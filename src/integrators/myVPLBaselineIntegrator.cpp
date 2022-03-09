#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include "nanoflann.hpp"
#include "anisovplq.h"

#include "igl/principal_curvature.h"
#include <igl/invert_diag.h>
#include "igl/list_to_matrix.h"

#include <igl/adjacency_list.h>

#include <unordered_map>

using namespace nanoflann;
using namespace std;

MTS_NAMESPACE_BEGIN

class MyVPLBaselineIntegrator : public SamplingIntegrator
{
public:
    MTS_DECLARE_CLASS()

    string rendermodeNames[4] = {"VPLs", "Clusters","Photons", "Aniso"};
    string samplingTechniqueNames[3] = {"uniform", "importance", "ris"};
    string bsdfTechNames[2] = {"bsdf", "diffuse"};
    function<float(Disk)> clusterImportanceFunction = [](Disk arg){return arg.totalEmission.max() / (float) arg.numVPLs;};
    function<float(VPL)> vplImportanceFunction = [](VPL arg){return arg.L_i.max();};
    #define VPLs 0
    #define CLUSTERS 1
    #define PHOTON 2
    #define ANISO 3

    #define UNIFORM 0
    #define IMPORTANCE 1    
    #define RIS 2

    #define BSDF 0
    #define DIFFUSE_APPROX 1

    MyVPLBaselineIntegrator(const Properties &props) : SamplingIntegrator(props)
    {
        sceneProperty_directIlluminationSamples = props.getInteger("directIlluminationSamples", 0);
        sceneProperty_indirectIlluminationSamples = props.getInteger("indirectIlluminationSamples", 0);
        sceneProperty_VPLCount = props.getInteger("desiredVPLCount", 0);
        sceneProperty_maxVPLPathLength = props.getInteger("maxPathLength", 1);
        sceneProperty_clampDistanceMin = props.getFloat("clampDistanceTo", 1.0f);
        sceneProperty_numberOfClusters = props.getInteger("numberOfClusters", 10);
        sceneProperty_refinementIterations = props.getInteger("refinementIterations", 10);
        sceneProperty_debug_showClusters = props.getInteger("debug_showClusters", 0) > 0;
        sceneProperty_ris_m = props.getInteger("ris_m", 1);
        sceneProperty_noShadows = props.getInteger("ignoreShadowRays", 0) > 0;
        sceneProperty_renderMode = props.getInteger("renderMode", 0);
        sceneProperty_distanceMetricWeight_flux = props.getFloat("distanceMetricWeight_flux", 1.0f);
        sceneProperty_distanceMetricWeight_distance = props.getFloat("distanceMetricWeight_distance", 1.0f);
        sceneProperty_distanceMetricWeight_normals = props.getFloat("distanceMetricWeight_normals", 1.0f);
        sceneProperty_clusterGenerationProjection = props.getInteger("clusterGenerationProjection", 0) > 0;
        sceneProperty_samplingTechnique = props.getInteger("samplingTechnique", 0);
        sceneProperty_bsdfMode = props.getInteger("bsdfMode", 0);
        sceneProperty_photonMappingRadiusN = props.getInteger("photonN", 10);
        sceneProperty_photonRadius = props.getFloat("photonRadius", 1.0f);
    }

    MyVPLBaselineIntegrator(Stream *stream, InstanceManager *manager) : SamplingIntegrator(stream, manager)
    {
        printf("TODO: Implement deserialization for my VPL integrator!\n");
    }

    void serialize(Stream *stream, InstanceManager *manager) const
    {
        SamplingIntegrator::serialize(stream, manager);
        printf("TODO: Implement serialization for my VPL integrator!\n");
    }

    void calculateEllipsoids(const Scene *scene)
    {
        using namespace Eigen;
        for (TriMesh* tmesh : scene->getMeshes())
        {
            if (tmesh->getTriangleCount() < 3)
                continue;
            vertexAttribOffset.push_back(perVertexPrincipalDir1.size());
            triMeshPointers.push_back(tmesh);
            
            vector<Point> vertexPts;
            std::vector<std::vector<float>> preV;
            std::vector<std::vector<int>> preF;
            
            unordered_map<int,int> indexMap;

            Point* vertices = tmesh->getVertexPositions();
            Triangle* triangles = tmesh->getTriangles();
            for (int t = 0; t < tmesh->getTriangleCount(); t++)
            {
                Point triangleVertices[3] = {vertices[triangles[t].idx[0]], vertices[triangles[t].idx[1]], vertices[triangles[t].idx[2]]};
                vector<int> face;
                for (int p = 0; p < 3; p++)
                {
                    int indexInVertexVec = vertexPts.size();
                    for (int i = 0; i < vertexPts.size(); i++)
                    {
                        if (distance(vertexPts[i], triangleVertices[p]) < 0.001f)
                        {
                            indexInVertexVec = i;
                            break;
                        }
                    }
                    if (indexInVertexVec >= vertexPts.size())
                    {
                        vertexPts.push_back(triangleVertices[p]);
                        face.push_back(vertexPts.size()-1);
                    } else {
                        face.push_back(indexInVertexVec);
                    }
                    indexMap[p] = indexInVertexVec;
                }
                preF.push_back(face);
            }
            printf(">>> shrunk vertex buffer from %i to %i\n", tmesh->getTriangleCount(), vertexPts.size());
            for (Point p : vertexPts)
            {
                preV.push_back(vector<float>({p.x, p.y, p.z}));
            }
            Eigen::MatrixXd V;
            Eigen::MatrixXi F;
            igl::list_to_matrix(preV, V);
            igl::list_to_matrix(preF, F);
            vector<vector<int>> vertex_to_vertices;
            igl::adjacency_list(F, vertex_to_vertices);
            int mostAdj = 0;
            int leastAdj = vertex_to_vertices.size();
            for (int i = 0; i < vertex_to_vertices.size(); i++)
            {
                mostAdj = max(mostAdj, (int) vertex_to_vertices[i].size());
                leastAdj = min(leastAdj, (int)vertex_to_vertices[i].size());
            }
            MatrixXd PD1, PD2;
            VectorXd PV1, PV2;
            vector<int> badVerts;
            igl::principal_curvature(V, F, PD1, PD2, PV1, PV2,badVerts);
            VectorXd H = 0.5 * (PV1 + PV2);
            Normal *normals = tmesh->getVertexNormals();
            for (int v = 0; v < tmesh->getVertexCount(); v++)
            {
                Point vertex = vertices[v];
                for (int i = 0; i < vertexPts.size(); i++)
                {
                    if (distance(vertexPts[i], vertex) < 0.001f)
                    {
                        perVertexPrincipalDir1.push_back(Vector(PD1.row(i)[0], PD1.row(i)[1], PD1.row(i)[2]));
                        perVertexPrincipalDir2.push_back(Vector(PD2.row(i)[0], PD2.row(i)[1], PD2.row(i)[2]));
                        perVertexCurvatureMagnitudes.push_back(Vector2f(PV1[i], PV2[i]));
                        break;
                    }
                }
            }
        }

       printf("Ellipsoids calculated, attribvector length: %i\n", perVertexPrincipalDir1.size());
    }
    

    Disk clusterToDisk(std::vector<VPL> vpls)
    {
        if (vpls.size() < 1)
        {
            return Disk { Point(FLT_MAX, FLT_MAX, FLT_MAX), Normal(1.0f, 0.0f, 0.0f), Spectrum(0.0f), 0.0f ,0};
        }
        Point center = std::accumulate(vpls.begin(), vpls.end(), Point(0.0f), [](Point acc, VPL arg) { return acc + arg.its.p; }) /(float)vpls.size();
        Vector n = std::accumulate(vpls.begin(), vpls.end(), Vector(0.0f), [](Vector acc, VPL arg)
                                   { return acc + arg.its.geoFrame.n; }) /
                   (float)vpls.size();
        n = n / n.length();
        float radius = std::accumulate(vpls.begin(), vpls.end(), 0.0f, [center](float acc, VPL arg)
                                       { return std::max(acc, distance(arg.its.p, center)); });
        Spectrum totalEmission = std::accumulate(vpls.begin(), vpls.end(), Spectrum(0.0f), [this](Spectrum acc, VPL arg)
                                                 { return acc + diffuseVPLEmission(arg); });
        std::vector<Vector> projectedRelativePositions;
        float maxDistToPlane = 0;
        for (VPL vpl : vpls)
        {
            Vector relPos = vpl.its.p - center;
            Vector projRelPos = relPos - dot(n, relPos) * relPos;
            projectedRelativePositions.push_back(projRelPos);
            maxDistToPlane = max(maxDistToPlane, dot(n, relPos - projRelPos)); // should be equivalent to: len(rp-prp) if len(rp-prp) > 0 else 0
        }
        // radius = max length of projected relativePositions
        // center += max length projetedRelativePosition-relativePosition
        if (!sceneProperty_clusterGenerationProjection)
            maxDistToPlane = 0.0f;
        Disk result = {center + (Epsilon + maxDistToPlane) * n, Normal(n), totalEmission, max(0.001f,radius), (int)vpls.size()}; // TODO: remove magic number radius
        return result;
    }

    int sampleByDistribution(float rand, std::vector<float> accWeights, float* pdfVal) const
    {
        int low = 0;
        int high = (int)accWeights.size() - 1;
        while (high - low > 1)
        {
            int mid = (low + high) / 2;
            if (accWeights[mid] > rand)
            {
                high = mid;
            }
            else
            {
                low = mid;
            }
        }
        int idx = accWeights[low] > rand ? low : high;
        *pdfVal = idx == 0 ? accWeights[idx] : accWeights[idx] - accWeights[idx-1];
        return idx;
    }

    std::vector<Disk> generateVPLClusters(std::vector<VPL> vpls, int numClustersToGen)
    {
        return std::vector<Disk>();
    }

    void printInfo()
    {
        printf("\n===\n");
        printf("\nRendering: %s %s %s\n", rendermodeNames[sceneProperty_renderMode].c_str(), samplingTechniqueNames[sceneProperty_samplingTechnique].c_str(), bsdfTechNames[sceneProperty_bsdfMode].c_str());
        if (sceneProperty_debug_showClusters)
        {
            printf("Visualizing clusters\n");
        }
        printf("\n===\n");
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int cameraResID, int samplerResID)
    {
        SamplingIntegrator::preprocess(scene, queue, job, sceneResID, cameraResID, samplerResID);
        static Sampler *sampler = NULL;
        if (!sampler)
        {
            Properties props("independent");
            sampler = static_cast<Sampler *>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), props));
            sampler->configure();
            sampler->generate(Point2i());
        }
        // VPL generation
        printf("Generating VPLs...\n");
        size_t offset = 5;
        generatedPaths = 0;
        for (int i = 0; i < 1000000 && vplVec.size() < sceneProperty_VPLCount; i++, generatedPaths++)
        {
            sampler->setSampleIndex(++offset);
            PositionSamplingRecord pRecOnEmitter;
            Spectrum pWeight = scene->sampleEmitterPosition(pRecOnEmitter, sampler->next2D());
            const Emitter *emitter = static_cast<const Emitter *>(pRecOnEmitter.object);
            DirectionSamplingRecord dRec;
            Spectrum dWeight = emitter->sampleDirection(dRec, pRecOnEmitter, sampler->next2D());
            Spectrum currentPathFlux = pWeight * dWeight;
            RayDifferential ray(pRecOnEmitter.p, dRec.d, 0);
            Intersection its;
            float albedo = 1;
            int pathDepth = 0;
            while (sampler->next1D() > 1 - albedo && scene->rayIntersect(ray, its) && pathDepth < sceneProperty_maxVPLPathLength)
            {
                currentPathFlux /= albedo;
                VPL v = {its,-ray.d,currentPathFlux,its.getBSDF(ray),pathDepth};
                vplVec.push_back(v);
                BSDFSamplingRecord nextBRec(its, sampler, EImportance);
                Spectrum cBsdfVal = its.getBSDF()->sample(nextBRec, sampler->next2D());
                ray = Ray(its.p, its.toWorld(nextBRec.wo), 0.0f);
                currentPathFlux *= cBsdfVal;
                albedo = std::min(0.95f, cBsdfVal.max());
                pathDepth++;
                if (cBsdfVal.isZero())
                    break;
            }
        }
        for (int i = 0; i < vplVec.size(); i++)
        {
            vplVec[i].L_i /= (float)generatedPaths;
        }
        // Clustering
        if (sceneProperty_debug_showClusters || sceneProperty_renderMode == CLUSTERS || sceneProperty_renderMode == ANISO)
        {
            float w_dist = sceneProperty_distanceMetricWeight_distance, w_n = sceneProperty_distanceMetricWeight_normals, w_flux = sceneProperty_distanceMetricWeight_flux;
            std::function<float(Disk, VPL)> clusterAndVPLdistanceMetric = [this, w_dist, w_n, w_flux](const Disk &cluster, const VPL &vpl)
            {
                Vector rgb1, rgb2;
                cluster.totalEmission.toLinearRGB(rgb1[0], rgb1[1], rgb1[2]);
                diffuseVPLEmission(vpl).toLinearRGB(rgb2[0], rgb2[1], rgb2[2]);
                rgb1 /= rgb1.length() > Epsilon ? rgb1.length() : 1;
                rgb2 /= rgb2.length() > Epsilon ? rgb2.length() : 1;
                return w_dist * distance(cluster.center, vpl.its.p) + w_n * (1 - dot(cluster.n, vpl.its.geoFrame.n)) + w_flux * dot(rgb2 - rgb1, rgb2 - rgb1);
            };
            std::function<Disk(VPL)> vplToCluster = [this](const VPL arg)
            {
                Spectrum emission = diffuseVPLEmission(arg);
                Disk c = {arg.its.p, arg.its.geoFrame.n, emission, 1.0f, 1};
                return c;
            };
            // Clustering init
            printf("Initializing Clustering\n");
            std::set<int> clusterInits;
            clusterVec.clear();
            std::vector<Disk> clustersIn;
            std::vector<Disk> clustersOut; // TODO: use this
            if (sceneProperty_numberOfClusters == sceneProperty_VPLCount)
            {
                std::transform(vplVec.begin(), vplVec.end(), std::back_inserter(clustersIn), vplToCluster);
            }
            else
            {
                int idx = (int)(sampler->next1D() * vplVec.size());
                clusterInits.insert(std::min((int)(sampler->next1D() * vplVec.size()), (int)vplVec.size() - 1));
                for (int i = 0; i < std::min(10, sceneProperty_numberOfClusters); i++)
                {
                    std::vector<float> differenceVec;
                    std::function<float(VPL, VPL)> metric = [](const VPL &arg1, const VPL &arg2)
                    { return distance(arg1.its.p, arg2.its.p); };
                    std::transform(vplVec.begin(), vplVec.end(), std::back_inserter(differenceVec), [this, clusterInits, metric](VPL arg)
                                   { return std::accumulate(clusterInits.begin(), clusterInits.end(), FLT_MAX,
                                                            [this, arg, metric](float acc, int s)
                                                            { return std::min(acc, metric(vplVec[s], arg)); }); });
                    int mostDifferent = (int)distance(differenceVec.begin(), std::max_element(differenceVec.begin(), differenceVec.end()));
                    clusterInits.insert(mostDifferent);
                }
                while (clusterInits.size() < sceneProperty_numberOfClusters)
                {
                    int idx = (int)(sampler->next1D() * vplVec.size());
                    clusterInits.insert(std::min(idx, (int)vplVec.size() - 1));
                }
                std::transform(clusterInits.begin(), clusterInits.end(), std::back_inserter(clustersIn), [this, vplToCluster](const int arg)
                               { return vplToCluster(vplVec[arg]); });
            }
            // Clustering
            printf("Clustering...\n");
            std::vector<int> clusterAssignments(vplVec.size(), -1);
            int refinementIteration = 0;
            std::vector<int> clusterSizes(clustersIn.size(), 0);
            for (int k = 0; k < sceneProperty_refinementIterations + 1 && !(sceneProperty_numberOfClusters == sceneProperty_VPLCount && k > 0); k++)
            {
                clusterSizes = std::vector<int>(clustersIn.size(), 0);
                for (int i = 0; i < vplVec.size(); i++)
                {
                    std::vector<Disk>::iterator minCluster = std::min_element(clustersIn.begin(), clustersIn.end(), [this, i, clusterAndVPLdistanceMetric](Disk arg1, Disk arg2)
                                                                              { return clusterAndVPLdistanceMetric(arg1, vplVec[i]) < clusterAndVPLdistanceMetric(arg2, vplVec[i]); });
                    int minClusterIdx = (int)std::distance(clustersIn.begin(), minCluster);
                    clusterAssignments[i] = minClusterIdx;
                    clusterSizes[minClusterIdx] += 1;
                }
                for (int i = 0; i < clustersIn.size(); i++)
                {
                    std::vector<VPL> vpls;
                    for (int j = 0; j < clusterAssignments.size(); j++)
                    {
                        if (clusterAssignments[j] == i)
                            vpls.push_back(vplVec[j]);
                    }
                    clustersIn[i] = clusterToDisk(vpls);
                }
                refinementIteration++;
            }
            clusterVec = clustersIn;
            printf("VPL: %zi / %i in %i paths \n", vplVec.size(), sceneProperty_VPLCount, generatedPaths);
            float avgR = std::accumulate(clusterVec.begin(), clusterVec.end(), 0.0f, [](float acc, Disk arg)
                                         { return acc + arg.r; }) /
                         (float)clusterVec.size();
            float minR = std::accumulate(clusterVec.begin(), clusterVec.end(), FLT_MAX, [](float acc, Disk arg)
                                         { return std::min(acc, arg.r); });
            float maxR = std::accumulate(clusterVec.begin(), clusterVec.end(), 0.0f, [](float acc, Disk arg)
                                         { return std::max(acc, arg.r); });
            int minClusterSize = *std::min_element(clusterSizes.begin(), clusterSizes.end());
            int maxClusterSize = *std::max_element(clusterSizes.begin(), clusterSizes.end());
            int totalVPLsClustered = std::accumulate(clusterSizes.begin(), clusterSizes.end(), 0, [](int acc, int arg)
                                                     { return acc + arg; });
            Vector rgbV, rgbC;
            std::accumulate(vplVec.begin(), vplVec.end(), Spectrum(0.0f), [this](Spectrum acc, VPL arg)
                            { return acc + diffuseVPLEmission(arg); })
                .toLinearRGB(rgbV[0], rgbV[1], rgbV[2]);
            std::accumulate(clusterVec.begin(), clusterVec.end(), Spectrum(0.0f), [](Spectrum acc, Disk arg)
                            { return acc + arg.totalEmission; })
                .toLinearRGB(rgbC[0], rgbC[1], rgbC[2]);
            printf("Emission: (%f,%f,%f),(%f,%f,%f)\n", rgbV[0], rgbV[1], rgbV[2], rgbC[0], rgbC[1], rgbC[2]);
            printf("Clusters: %i, size: %i to %i, r: %f to %f, avg_r %f\n", (int)clusterVec.size(),
                   *std::min_element(clusterSizes.begin(), clusterSizes.end()),
                   *std::max_element(clusterSizes.begin(), clusterSizes.end()), minR, maxR, avgR);
            printf("Clustered %i vpls\n", totalVPLsClustered);
            std::vector<Disk> preClusterVec(clusterVec);
            clusterVec.clear();
            std::copy_if(preClusterVec.begin(), preClusterVec.end(), std::back_inserter(clusterVec), [](Disk arg)
                         { return abs(arg.n.length() - 1.0f) < Epsilon && arg.totalEmission.max() > Epsilon; });
            printf("Removed %zd invalid clusters\n", preClusterVec.size() - clusterVec.size());
        } else {
            printf("Skipped Clustering\n");
        }

        // accumulate total flux for importance sampling vpl/clusters
        float currentWeightPrefixSum = 0;
        for (int i = 0; i < clusterVec.size(); i++)
        {
            currentWeightPrefixSum += clusterImportanceFunction(clusterVec[i]);
            clusterWeightPrefixVec.push_back(currentWeightPrefixSum);
        }
        transform(clusterWeightPrefixVec.begin(), clusterWeightPrefixVec.end(), clusterWeightPrefixVec.begin(), [currentWeightPrefixSum](float arg) { return arg / currentWeightPrefixSum; });
        currentWeightPrefixSum = 0;
        for (int i = 0; i < vplVec.size(); i++)
        {
            currentWeightPrefixSum += vplImportanceFunction(vplVec[i]);
            vplWeightPrefixVec.push_back(currentWeightPrefixSum);
        }
        transform(vplWeightPrefixVec.begin(), vplWeightPrefixVec.end(), vplWeightPrefixVec.begin(), [currentWeightPrefixSum](float arg) { return arg / currentWeightPrefixSum; });
        // build vpl kd-tree
        transform(vplVec.begin(), vplVec.end(), std::back_inserter(cloud.pts), [](VPL arg) { return arg.its.p; });
        index = new KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, VPLCloud>, VPLCloud, 3>(3, cloud, KDTreeSingleIndexAdaptorParams(10));
        index->buildIndex();
        if (sceneProperty_renderMode == ANISO)
        {
            printf("Calculating Ellipsoids...\n");
            calculateEllipsoids(scene);
            Vector2f curv = accumulate(perVertexCurvatureMagnitudes.begin(), perVertexCurvatureMagnitudes.end(), Vector2f(FLT_MIN, FLT_MAX), [](Vector2f acc, Vector2f arg)
                                       { return Vector2f(max(acc.x, arg.x), min(acc.y, arg.y)); });
            printf("curvature: [%f,%f]\n", curv.x, curv.y);
        }

        sceneDistanceUpperBound = 2.0f * scene->getAABB().getBSphere().radius;
        printInfo();
        Vector2 gridSize = Vector2(15, 15);
        for (int i = 0; i < gridSize.x; i++)
        {
            for (int j = 0; j < gridSize.y; j++)
            {
                Ray mainRay;
                Vector2i res = scene->getSensor()->getFilm()->getSize();
                Point2 relPt = Point2(((float)i) / gridSize.x, ((float)j) / gridSize.y);
                relPt = 0.25f * relPt + Point2(0.35f,0.5f);
                scene->getSensor()->sampleRay(mainRay, Point2(relPt.x * res.x, relPt.y * res.y), Point2(0.5f, 0.5f), 0);
                Intersection its;
                //printf("--> (%f,%f,%f) __ (%f,%f)\n", mainRay.d.x, mainRay.d.y, mainRay.d.z, relPt.x, relPt.y);
                if (scene->rayIntersect(mainRay, its))
                {
                    debug_isects.push_back(its);
                }
            }
        }
        //printf("huhu: %i\n", debug_isects.size());
        return true;
    }

    Spectrum evalVPLContribution(const RayDifferential &r, VPL vpl, RadianceQueryRecord &rRec, bool shadowRay) const
    {
        Vector toVPL = Vector(vpl.its.p - rRec.its.p);
        Vector d = toVPL / toVPL.length();
        RayDifferential toVPLrd(rRec.its.p + Epsilon * d, d, 0);
        Intersection its;
        if (dot(d, rRec.its.geoFrame.n) < Epsilon || dot(-toVPLrd.d, vpl.its.geoFrame.n) < Epsilon)
            return Spectrum(0.0f);
        if (shadowRay)
        {
            rRec.scene->rayIntersect(toVPLrd, its);
        }
        if (!shadowRay || its.t + ShadowEpsilon >= toVPL.length())
        {
            BSDFSamplingRecord bRec_x2(vpl.its, vpl.its.toLocal(-toVPLrd.d), vpl.its.toLocal(vpl.w_o));
            BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(toVPLrd.d));
            Spectrum f_x2 = vpl.f->eval(bRec_x2) / bRec_x2.wo.z;
            //f_x2 = std::max(0.0f,dot(vpl.its.geoFrame.n,vpl.w_o)) * vpl.f->getDiffuseReflectance(vpl.its) / M_PI;
            Spectrum f_x1 = rRec.its.getBSDF(r)->eval(bRec_x1);
            Float geo_yx = std::max(dot(-d, Vector(vpl.its.geoFrame.n)), 0.0f);
            Float clampedDistance = std::max(toVPL.length(), sceneProperty_clampDistanceMin);
            Float geometryTerm = geo_yx / pow(clampedDistance, 2.0f);
            return f_x1 * geometryTerm * f_x2 * vpl.L_i;
        }
        return Spectrum(0.0f);
    }

    Spectrum evalClusterContribution(const RayDifferential &r, Disk c, Point3 p,RadianceQueryRecord &rRec, bool shootShadowRay) const
    {
        Vector toLight(p - rRec.its.p);
        Vector d = toLight / toLight.length();
        RayDifferential rayDiff(rRec.its.p + Epsilon * d, d, 0);
        Intersection its;
        if (shootShadowRay)
            rRec.scene->rayIntersect(rayDiff, its);
        if (!shootShadowRay || its.t + ShadowEpsilon >= toLight.length())
        {
            BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(d));
            Float clampedDistance = std::max(toLight.length(), sceneProperty_clampDistanceMin);
            Float geometryTerm = std::max(dot(-d, Vector(c.n)), 0.0f) / pow(clampedDistance, 2.0f);
            return rRec.its.getBSDF(r)->eval(bRec_x1) * geometryTerm * c.totalEmission;
        }
        return Spectrum(0.0f);
    }

    Spectrum debug_showClusters(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        Disk bestC;
        Spectrum color = rRec.rayIntersect(r) ? rRec.its.getBSDF(r)->getDiffuseReflectance(rRec.its) / M_PI: Spectrum(0.0f);
        std::vector<float> tVec;
        transform(clusterVec.begin(), clusterVec.end(), std::back_inserter(tVec), [r](Disk arg){
            float t = FLT_MAX;
            if (intersectDisk(r,arg, &t))
                return t;
            return FLT_MAX; 
        });
        size_t minIdx = distance(tVec.begin(), std::min_element(tVec.begin(), tVec.end()));
        bestC = clusterVec[minIdx];
        if (tVec[minIdx] < rRec.its.t + Epsilon)
        {
            Spectrum spec = clusterVec[minIdx].totalEmission / (M_PI * pow(clusterVec[minIdx].r,2.0f));
            return spec;
        } else {
            DirectSamplingRecord dRec(rRec.its);
            Spectrum value = rRec.scene->sampleEmitterDirect(dRec, rRec.sampler->next2D());
            BSDFSamplingRecord bRec(rRec.its, -rRec.its.toLocal(r.d), rRec.its.toLocal(dRec.d));
            Spectrum bsdfVal = rRec.its.getBSDF(r)->eval(bRec);
            return value * bsdfVal;
        }
    }
    
    Spectrum debug_showVPLs(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        size_t num_results = 1;
        std::vector<uint32_t> ret_index(num_results);
        std::vector<float> out_dist_sqr(num_results);
        const float query_pt[3] = {rRec.its.p[0], rRec.its.p[1], rRec.its.p[2]};
        num_results = index->knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        if (num_results > 0)
        {
            float distToClosestVPL = sqrt(out_dist_sqr[0]);
            if (distToClosestVPL < 0.00001)
            {
                return Spectrum(1.0f);
            }
            return Spectrum(1.0f / distToClosestVPL);
        }
        return Spectrum(0.0f);
    }

    Spectrum photonQuery(const RayDifferential &ray, Intersection isect, RadianceQueryRecord &rRec, int photonN) const
    {
        size_t num_results = photonN;
        std::vector<uint32_t> ret_index(num_results);
        std::vector<float> out_dist_sqr(num_results);
        const float query_pt[3] = {isect.p[0], isect.p[1], isect.p[2]};
        num_results = index->knnSearch(&query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        Spectrum L_o(0.0f);
        for (int p = 0; p < num_results; p++)
        {
            float radiusSquared = *std::max_element(out_dist_sqr.begin(), out_dist_sqr.end());
            VPL vpl = vplVec[ret_index[p]];
            BSDFSamplingRecord bRec_x2(isect, isect.toLocal(-ray.d), isect.toLocal(vpl.w_o));
            if(bRec_x2.wo.z < Epsilon)
                continue;
            Spectrum f_x2 = isect.getBSDF()->eval(bRec_x2) / bRec_x2.wo.z;
            L_o += (f_x2 * vpl.L_i) / (M_PI * radiusSquared);
        }
        return L_o;
    }

    bool getLocalEllipse(const Intersection isect, Vector &dir, float &length, float &width) const
    {
        size_t dist = distance(triMeshPointers.begin(), find(triMeshPointers.begin(), triMeshPointers.end(), isect.shape));
        if (dist < triMeshPointers.size())
        {
            const TriMesh *tmesh = static_cast<const TriMesh *>(isect.shape);
            Triangle tri = tmesh->getTriangles()[isect.primIndex];
            Point v0 = tmesh->getVertexPositions()[tri.idx[0]];
            Point v1 = tmesh->getVertexPositions()[tri.idx[1]];
            Point v2 = tmesh->getVertexPositions()[tri.idx[2]];
            Vector e01 = v1 - v0;
            Vector e02 = v2 - v0;
            float totalA = cross(e01, e02).length();
            float a0 = cross(v1 - isect.p, v2 - isect.p).length() / totalA;
            float a1 = cross(v0 - isect.p, v2 - isect.p).length() / totalA;
            float a2 = 1 - a0 - a1;
            Vector values[6] = {
                perVertexPrincipalDir1[vertexAttribOffset[dist] + tri.idx[0]],
                perVertexPrincipalDir1[vertexAttribOffset[dist] + tri.idx[1]],
                perVertexPrincipalDir1[vertexAttribOffset[dist] + tri.idx[2]],
                perVertexPrincipalDir2[vertexAttribOffset[dist] + tri.idx[0]],
                perVertexPrincipalDir2[vertexAttribOffset[dist] + tri.idx[1]],
                perVertexPrincipalDir2[vertexAttribOffset[dist] + tri.idx[2]],
            };
            Vector2f mag0 = perVertexCurvatureMagnitudes[vertexAttribOffset[dist] + tri.idx[0]];
            Vector2f mag1 = perVertexCurvatureMagnitudes[vertexAttribOffset[dist] + tri.idx[1]];
            Vector2f mag2 = perVertexCurvatureMagnitudes[vertexAttribOffset[dist] + tri.idx[2]];

            Vector dir1_ipol, dir2_ipol;
            Vector2f ipol_mag;
            if (distance(v0, isect.p) < distance(v1, isect.p) && distance(v0, isect.p) < distance(v2, isect.p))
            {
                dir1_ipol = values[0];
                dir2_ipol = values[3];
                ipol_mag = mag0;
            }
            else if (distance(v1, isect.p) < distance(v0, isect.p) && distance(v1, isect.p) < distance(v2, isect.p))
            {
                dir1_ipol = values[1];
                dir2_ipol = values[4];
                ipol_mag = mag1;
            }
            else
            {
                dir1_ipol = values[2];
                dir2_ipol = values[5];
                ipol_mag = mag2;
            }
            ipol_mag = a0 * mag0 + a1 * mag1 + a2 * mag2;

            if (ipol_mag.x > 0 && ipol_mag.y > 0) // convex
            {
                length = 1.0 / ipol_mag.x;
                width = 1.0 / ipol_mag.y;
                dir = dir1_ipol;
            } else if (ipol_mag.x < 0 && ipol_mag.y < 0) // concave
            {
                length = 1.0 / ipol_mag.y;
                width = 1.0 / ipol_mag.x;
                dir = dir2_ipol;
            } else
            {
                if (ipol_mag.x > -ipol_mag.y)
                {
                    length = 1.0 / ipol_mag.x;
                    width = 1.0 / -ipol_mag.y;
                    dir = dir1_ipol;
                } else {
                    length = 1.0 / -ipol_mag.y;
                    width = 1.0 / ipol_mag.x;
                    dir = dir2_ipol;
                }
            }
            length *= 0.05;
            width *= 0.05;
            length = min(length,sceneProperty_photonRadius);
            width = min(width,sceneProperty_photonRadius);
            return true;
        }
        length = sceneProperty_photonRadius;
        width = sceneProperty_photonRadius;
        dir = isect.geoFrame.t;
        return true;
    }

    Spectrum ellipsoidQuery(const RayDifferential &ray, Intersection isect, RadianceQueryRecord &rRec, Vector principalDirection, float length, float width) const
    {   
        Vector ellLongAxis = principalDirection;
        Vector ellShortAxis = cross(isect.geoFrame.n, principalDirection);

        const float search_radius = length;
        std::vector<std::pair<uint32_t, float>> ret_matches;
        nanoflann::SearchParams params;
        const float query_pt[3] = {isect.p[0], isect.p[1], isect.p[2]};
        const size_t nMatches = index->radiusSearch(&query_pt[0], search_radius, ret_matches, params);

        Spectrum L_o(0.0f);
        for (int p = 0; p < nMatches; p++)
        {
            VPL vpl = vplVec[ret_matches[p].first];
            if (!isInsideEllipse(isect.p, ellLongAxis, ellShortAxis, length, width, vpl.its.p))
            {
                continue;
            }
            BSDFSamplingRecord bRec_x2(isect, isect.toLocal(-ray.d), isect.toLocal(vpl.w_o));
            if (bRec_x2.wo.z < Epsilon)
                continue;
            Spectrum f_x2 = isect.getBSDF()->eval(bRec_x2) / bRec_x2.wo.z;
            L_o += (f_x2 * vpl.L_i) / (length*width * M_PI);
        }
        return L_o;
    }

    Spectrum photonEllipsoidProcedure(const RayDifferential &ray, Intersection isect, RadianceQueryRecord &rRec) const
    {
        Vector dir;
        float first,second;
        if (getLocalEllipse(isect,dir,first,second))
        {
            return ellipsoidQuery(ray, isect, rRec, dir, first, second);
        } else {
            Vector dirInPlane = abs(isect.geoFrame.n.x) > 1 - Epsilon ? cross(Vector(0.0f, 1.0f, 0.0f), isect.geoFrame.n) : cross(Vector(1.0f, 0.0f, 0.0f), isect.geoFrame.n);
            return ellipsoidQuery(ray, isect, rRec, dirInPlane, sceneProperty_photonRadius, sceneProperty_photonRadius);
        }
    }

    bool visualTest(const RadianceQueryRecord &rRec, Spectrum &spec) const
    {
        bool isInsideAnyEllipse = false;
        function<float(float)> fract = [](float arg){return (float) (arg - (long) arg);};
        for (int i = 0; i < debug_isects.size(); i++)
        {
            Intersection its = debug_isects[i];
            Vector lengthDir;
            float first, second;
            if (distance(debug_isects[i].p,rRec.its.p) > sceneProperty_photonRadius)
            {
                continue;
            }
            if (getLocalEllipse(its, lengthDir, first, second))
            {
                spec.fromLinearRGB(1.0f,0.0f,0.0f);
                return true;
                Vector widthDir = normalize(cross(its.geoFrame.n, lengthDir));
                if (isInsideEllipse(rRec.its.p, lengthDir, widthDir, first, second, its.p))
                {
                    Vector2 vec = Vector2(debug_isects[i].p.x, debug_isects[i].p.y);
                    Vector rgb = Vector(abs(fract(sin(dot(vec, Vector2(12.9898f, 78.233f))) * 143.5453f)),
                                        abs(fract(sin(dot(vec, Vector2(5.9898, 112.233f))) * 223.5453f)),
                                        abs(fract(sin(dot(vec, Vector2(15.9898f, 50.233f))) * 5.5453f)));
                    rgb = normalize(rgb);
                    spec.fromLinearRGB(rgb.x, rgb.y, rgb.z);
                    return true;
                }
            }
        }
        return false;
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        
        if (rRec.rayIntersect(r))
        {
            Vector lengthDir;
            float first, second;
            getLocalEllipse(rRec.its,lengthDir,first,second);
            Spectrum spect;
            if (second < 0)
            {
                spect.fromLinearRGB(1.0f, 0.0f, 0.0f);
                return spect;
            }
            float val = 1.0f / second;
            
            spect.fromLinearRGB(val,val,val);
            return spect;
            /*
            Spectrum spec;
            if(visualTest(rRec,spec))
            {
                return spec;
            }
            */
        }
        
        Spectrum total(0.0f);
        if (rRec.rayIntersect(r))
        {
            if (sceneProperty_debug_showClusters)
            {
                return debug_showClusters(r, rRec);
            }
            if (rRec.its.isEmitter())
            {
                total += rRec.its.Le(-r.d);
            }
            DirectSamplingRecord dRec(rRec.its);
            for (int i = 0; i < sceneProperty_directIlluminationSamples; i++)
            {
                Spectrum value = rRec.scene->sampleEmitterDirect(dRec, rRec.sampler->next2D());
                BSDFSamplingRecord bRec(rRec.its, -rRec.its.toLocal(r.d), rRec.its.toLocal(dRec.d));
                Spectrum bsdfVal = rRec.its.getBSDF(r)->eval(bRec);
                Spectrum direct = value * bsdfVal;
                total += direct / (float)sceneProperty_directIlluminationSamples;
            }
            if (sceneProperty_indirectIlluminationSamples < 1)
                return total;
            Spectrum indirect = Spectrum(0.0f);
            if (sceneProperty_renderMode == VPLs)
            {
                if (sceneProperty_samplingTechnique == UNIFORM)
                {
                    for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                    {
                        indirect += evalVPLContribution(r, vplVec[uniformIndex(rRec.sampler->next1D(), vplVec.size())], rRec, !sceneProperty_noShadows) * (float)vplVec.size();
                    }
                    total += indirect / (float)sceneProperty_indirectIlluminationSamples;
                } else if (sceneProperty_samplingTechnique == IMPORTANCE) {
                    for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                    {
                        float pdfVal;
                        int idx = sampleByDistribution(rRec.sampler->next1D(), vplWeightPrefixVec, &pdfVal);
                        indirect += evalVPLContribution(r, vplVec[idx], rRec, !sceneProperty_noShadows) / pdfVal;
                    }
                    total += indirect / (float)sceneProperty_indirectIlluminationSamples;
                } else {
                    Spectrum spec;
                    spec.fromLinearRGB(1.0f,0.0f,0.0f);
                    return spec;
                }
            } else if (sceneProperty_renderMode == CLUSTERS) {
                if (sceneProperty_samplingTechnique == RIS)
                {
                    std::vector<Disk> elements(sceneProperty_ris_m, clusterVec[0]);
                    std::vector<float> weights(sceneProperty_ris_m, 0);
                    std::vector<float> gVals(sceneProperty_ris_m, 0);
                    std::vector<Point> pts(sceneProperty_ris_m, Point(0.0f));
                    for (int i = 0; i < sceneProperty_indirectIlluminationSamples; i++)
                    {
                        for (int j = 0; j < sceneProperty_ris_m; j++)
                        {
                            elements[j] = clusterVec[uniformIndex(rRec.sampler->next1D(), clusterVec.size())];
                            pts[j] = getPointOnDisk(elements[j], rRec.sampler->next2D());
                            Vector d = (pts[j] - rRec.its.p) / (pts[j] - rRec.its.p).length();
                            float gXjVal = elements[j].totalEmission.max() * std::max(dot(d, Vector(rRec.its.geoFrame.n)), 0.0f) * std::max(dot(-d, Vector(elements[j].n)), 0.0f) / pow(std::max((pts[j] - rRec.its.p).length(), sceneProperty_clampDistanceMin), 2.0f);
                            float pXjVal = 1.0f / (float)clusterVec.size();
                            weights[j] = pXjVal < Epsilon ? 0 : gXjVal / pXjVal;
                            gVals[j] = gXjVal;
                        }
                        float totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0f, [](float acc, float arg)
                                                            { return acc + arg; });
                        transform(weights.begin(), weights.end(), weights.begin(), [totalWeight](float arg)
                                  { return arg / totalWeight; });
                        float rand = rRec.sampler->next1D();
                        int idx = 0;
                        float prefixSum = 0;
                        for (int j = 0; j < elements.size() && prefixSum < rand; j++)
                        {
                            idx = j;
                            prefixSum += weights[j];
                        }
                        if (gVals[idx] > Epsilon)
                        {
                            indirect += evalClusterContribution(r, elements[idx], pts[idx], rRec, true && !sceneProperty_noShadows) / gVals[idx] * totalWeight / (float)elements.size();
                        }
                    }
                    total += indirect / (float)sceneProperty_indirectIlluminationSamples;
                } else if (sceneProperty_samplingTechnique == IMPORTANCE) {
                    for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                    {
                        float pdfVal;
                        int idx = sampleByDistribution(rRec.sampler->next1D(), clusterWeightPrefixVec, &pdfVal);
                        Disk c = clusterVec[idx];
                        indirect += evalClusterContribution(r, c, getPointOnDisk(c, rRec.sampler->next2D()), rRec, !sceneProperty_noShadows) / pdfVal;
                    }
                    total += indirect / (float)sceneProperty_indirectIlluminationSamples;
                } else { // uniform
                    for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                    {
                        Disk c = clusterVec[uniformIndex(rRec.sampler->next1D(), clusterVec.size())];
                        indirect += evalClusterContribution(r, c, getPointOnDisk(c, rRec.sampler->next2D()), rRec, !sceneProperty_noShadows);
                    }
                    total += indirect * (float)clusterVec.size() / (float)sceneProperty_indirectIlluminationSamples;
                }
            }
            else if (sceneProperty_renderMode == PHOTON)
            {
                if (sceneProperty_samplingTechnique == IMPORTANCE)
                {
                    for (int i = 0; i < sceneProperty_indirectIlluminationSamples; i++)
                    {
                        BSDFSamplingRecord nextBRec(rRec.its, rRec.sampler, EImportance);
                        Spectrum cBsdfVal = rRec.its.getBSDF()->sample(nextBRec, rRec.sampler->next2D());
                        RayDifferential ray;
                        ray = Ray(rRec.its.p, rRec.its.toWorld(nextBRec.wo), 0.0f);
                        Intersection its;
                        if (rRec.scene->rayIntersect(ray, its))
                        {
                            indirect += photonQuery(ray, its, rRec, sceneProperty_photonMappingRadiusN) * cBsdfVal;
                        }
                    }
                    total += indirect / (float)sceneProperty_indirectIlluminationSamples;
                } else if (sceneProperty_samplingTechnique == UNIFORM) {
                    for (int i = 0; i < sceneProperty_indirectIlluminationSamples; i++)
                    {
                        Vector local = warp::squareToUniformHemisphere(rRec.sampler->next2D());
                        Vector dir = Frame(rRec.its.geoFrame.n).toWorld(local);
                        float pdf = warp::squareToUniformHemispherePdf();
                        RayDifferential ray;
                        ray = Ray(rRec.its.p, dir, 0.0f);
                        Intersection its;
                        if (rRec.scene->rayIntersect(ray, its))
                        {
                            BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(ray.d));
                            Spectrum f_x1 = rRec.its.getBSDF(r)->eval(bRec_x1);
                            indirect += photonQuery(ray, its, rRec, sceneProperty_photonMappingRadiusN) * f_x1 / pdf;
                        }
                    }
                    total += indirect / (float)sceneProperty_indirectIlluminationSamples;
                } else {
                    for (int i = 0; i < sceneProperty_indirectIlluminationSamples; i++)
                    {
                        ref_vector<Shape> preShapes = rRec.scene->getShapes();
                        ref_vector<Shape> shapes;
                        copy_if(preShapes.begin(),preShapes.end(), back_inserter(shapes), [](Shape* arg){return !arg->isEmitter();});
                        size_t idx = uniformIndex(rRec.sampler->next1D(), shapes.size());
                        PositionSamplingRecord psr;
                        shapes[idx]->samplePosition(psr, rRec.sampler->next2D());
                        RayDifferential ray;
                        ray = Ray(rRec.its.p, normalize(Vector(psr.p - rRec.its.p)), 0.0f);
                        Intersection its;
                        if (rRec.scene->rayIntersect(ray, its))
                        {
                            BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(ray.d));
                            Spectrum f_x1 = rRec.its.getBSDF(r)->eval(bRec_x1);
                            float geo = max(0.0f, dot(-ray.d, its.geoFrame.n)) / pow(distance(its.p, ray.o), 2.0f);
                            float pdf = psr.pdf * (1.0f / (float)shapes.size());
                            indirect += photonQuery(ray, its, rRec, sceneProperty_photonMappingRadiusN) * f_x1 * geo / pdf;
                        }
                    }
                    total += indirect / (float)sceneProperty_indirectIlluminationSamples;
                }
            } else if (sceneProperty_renderMode == ANISO) 
            {
                for (int i = 0; i < sceneProperty_indirectIlluminationSamples; i++)
                {
                    //float pdfVal;
                    //Disk c = clusterVec[sampleByDistribution(rRec.sampler->next1D(), clusterWeightPrefixVec, &pdfVal)];

                    Disk c = clusterVec[uniformIndex(rRec.sampler->next1D(), clusterVec.size())];
                    float pdfVal = 1.0f / (float)clusterVec.size();

                    Point pt = getPointOnDisk(c, rRec.sampler->next2D());
                    pdfVal *= 1.0f / (M_PI * pow(c.r, 2.0f));
                    Intersection its;
                    RayDifferential ray;
                    ray = Ray(rRec.its.p + Epsilon * rRec.its.geoFrame.n, normalize(pt - rRec.its.p), 0.0f);
                    if (dot(ray.d,c.n) > -Epsilon) // cluster hit from back
                        continue;
                    if (distance(pt, rRec.its.p) < Epsilon || dot(ray.d, rRec.its.geoFrame.n) < Epsilon)
                        continue;
                    if (pointIsOnDisk(c, rRec.its.p)) // TODO: not supposed to continue, but account for pdf in mis
                        continue;
                    if (rRec.scene->rayIntersect(ray, its))
                    {
                        if (distance(rRec.its.p, its.p) > Epsilon && distance(its.p, c.center) < c.r) // TODO: better shadow test, distance to disk
                        {
                            float misTotal = 0.0f;
                            for (Disk disk : clusterVec)
                            {
                                float t;
                                if (intersectDisk(ray, disk,&t) && t <= its.t)
                                {
                                    float otherPdf = 1.0f / (M_PI * pow(c.r, 2.0f) * clusterVec.size());
                                    misTotal = otherPdf;
                                    // TODO needs special case if ray is in disk plane
                                }
                            }
                            if (misTotal == 0.0f)
                            {
                                misTotal = pdfVal;
                            }
                            BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(ray.d));
                            Spectrum f_x1 = rRec.its.getBSDF(r)->eval(bRec_x1);
                            //float geo = dot(-ray.d, its.geoFrame.n) / pow(max(sceneProperty_clampDistanceMin,distance(its.p, rRec.its.p)), 2.0f);
                            float geo = dot(-ray.d, its.geoFrame.n) / pow(distance(its.p, rRec.its.p), 2.0f);
                            //if (its.shape == rRec.its.shape) {continue;}
                            //indirect += (photonQuery(ray, its, rRec, sceneProperty_photonMappingRadiusN) * f_x1 * geo) / misTotal;
                            Vector ellDir;
                            float length,width;
                            getLocalEllipse(its,ellDir,length,width);
                            indirect += (ellipsoidQuery(ray, its, rRec, ellDir,length,width) * f_x1 * geo) / misTotal;
                            //indirect += Spectrum(dot(ray.d, rRec.its.geoFrame.n) * geo);
                        }
                    }
                }
                total += indirect / (float)sceneProperty_indirectIlluminationSamples;
            } else
            {
                printf("\n\nError: No valid mode set!\n\n");
                Spectrum spec;
                spec.fromLinearRGB(1.0f, 0.0f, 0.0f);
                return spec;
            }
        }
        return total;
    }

private:
    VPLCloud cloud;
    KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<float, VPLCloud>, VPLCloud, 3> *index;
    bool sceneProperty_debug_showClusters, sceneProperty_noShadows, sceneProperty_clusterGenerationProjection;
    int generatedPaths = 0;
    int sceneProperty_VPLCount, sceneProperty_directIlluminationSamples, sceneProperty_indirectIlluminationSamples, sceneProperty_maxVPLPathLength, sceneProperty_numberOfClusters, sceneProperty_refinementIterations, sceneProperty_ris_m,
        sceneProperty_bsdfMode, sceneProperty_samplingTechnique, sceneProperty_renderMode, sceneProperty_photonMappingRadiusN;
    float sceneProperty_clampDistanceMin, sceneProperty_distanceMetricWeight_flux, sceneProperty_distanceMetricWeight_distance, sceneProperty_distanceMetricWeight_normals, sceneProperty_photonRadius;
    float sceneDistanceUpperBound;
    std::vector<VPL> vplVec;
    std::vector<Disk> clusterVec;
    std::vector<float> clusterWeightPrefixVec, vplWeightPrefixVec;

    std::vector<Shape*> triMeshPointers;
    std::vector<int> vertexAttribOffset;
    std::vector<Vector> perVertexPrincipalDir1;
    std::vector<Vector> perVertexPrincipalDir2;
    std::vector<Vector2f> perVertexCurvatureMagnitudes;

    std::vector<Intersection> debug_isects;
};
MTS_IMPLEMENT_CLASS_S(MyVPLBaselineIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MyVPLBaselineIntegrator, "My Baseline VPL Integrator");

MTS_NAMESPACE_END