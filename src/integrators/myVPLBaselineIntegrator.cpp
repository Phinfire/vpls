#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>

#include <vector>
#include <algorithm>
#include <numeric>


MTS_NAMESPACE_BEGIN

class MyVPLBaselineIntegrator : public SamplingIntegrator
{
public:
    MTS_DECLARE_CLASS()

    struct VPL
    {
        Intersection its;
        Vector w_o;
        Spectrum L_i;
        const BSDF *f;
        int bounce;
    };

    struct VPLCluster
    {
        Point center;
        Normal n;
        Spectrum totalEmission;
        float r;
    };

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
        sceneProperty_useClusters = props.getInteger("useClusters", 0) != 0 ? true : false;
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

    Spectrum diffuseVPLEmission(const VPL &vpl) const
    {
        return vpl.L_i * vpl.f->getDiffuseReflectance(vpl.its) / M_PI;
    }

    void clusterVPLs(const Scene *scene, RenderQueue *queue, const RenderJob *job, int sceneResID, int cameraResID, int samplerResID)
    {

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
        size_t offset = 5;
        int generatedPaths = 0;
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
        float w_dist = 1, w_n = 1, w_flux = 1;
        std::function<float(VPLCluster, VPL)> clusterAndVPLdistanceMetric = [this,w_dist, w_n, w_flux](const VPLCluster &cluster, const VPL &vpl)
        {
            Vector rgb1, rgb2;
            cluster.totalEmission.toLinearRGB(rgb1[0], rgb1[1], rgb1[2]);
            diffuseVPLEmission(vpl).toLinearRGB(rgb2[0], rgb2[1], rgb2[2]);
            rgb1 /= rgb1.length() > Epsilon ? rgb1.length() : 1;
            rgb2 /= rgb2.length() > Epsilon ? rgb2.length() : 1;
            return w_dist * distance(cluster.center, vpl.its.p) + w_n * (1 - dot(cluster.n, vpl.its.geoFrame.n)) + w_flux * dot(rgb2 - rgb1, rgb2 - rgb1);
        };
        std::function<VPLCluster(VPL)> vplToCluster = [this](const VPL arg)
        {
            Spectrum emission = diffuseVPLEmission(arg);
            VPLCluster c = {arg.its.p, arg.its.geoFrame.n, emission, 1.0f};
            return c;
        };
        // Clustering init
        std::set<int> clusterInits;
        clusterVec.clear();
        if (sceneProperty_numberOfClusters == sceneProperty_VPLCount)
        {
            std::transform(vplVec.begin(), vplVec.end(), std::back_inserter(clusterVec), vplToCluster);
        } else {
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
                int mostDifferent = distance(differenceVec.begin(), std::max_element(differenceVec.begin(), differenceVec.end()));
                clusterInits.insert(mostDifferent);
            }
            while (clusterInits.size() < sceneProperty_numberOfClusters)
            {
                int idx = (int)(sampler->next1D() * vplVec.size());
                clusterInits.insert(std::min(idx, (int)vplVec.size() - 1));
            }
            std::transform(clusterInits.begin(), clusterInits.end(), std::back_inserter(clusterVec), [this,vplToCluster](const int arg){return vplToCluster(vplVec[arg]);});
        }
        // Clustering
        std::vector<int> clusterAssignments(vplVec.size(),-1);
        int refinementIteration = 0;
        std::vector<int> clusterSizes(clusterVec.size(), 0);
        for (int k = 0; k < sceneProperty_refinementIterations + 1; k++)
        {
            clusterSizes = std::vector<int>(clusterVec.size(), 0);
            for (int i = 0; i < vplVec.size(); i++)
            {
                std::vector<VPLCluster>::iterator minCluster = std::min_element(clusterVec.begin(), clusterVec.end(), [this, i, clusterAndVPLdistanceMetric](VPLCluster arg1, VPLCluster arg2)
                                                                           { return clusterAndVPLdistanceMetric(arg1, vplVec[i]) < clusterAndVPLdistanceMetric(arg2, vplVec[i]); });
                int minClusterIdx = (int) std::distance(clusterVec.begin(), minCluster);
                clusterAssignments[i] = minClusterIdx;
                clusterSizes[minClusterIdx] += 1;
            }
            for (int i = 0; i < clusterVec.size(); i++)
            {
                int clusterSize = std::count(clusterAssignments.begin(), clusterAssignments.end(), i);
                if (clusterSize > 0)
                {
                    std::vector<VPL> vpls;
                    for (int j = 0; j < clusterAssignments.size(); j++)
                    {
                        if (clusterAssignments[j] == i)
                            vpls.push_back(vplVec[j]);
                    }
                    Point center = std::accumulate(vpls.begin(), vpls.end(), Point(0.0f), [](Point acc, VPL arg)
                                                   { return acc + arg.its.p; }) /
                                   (float)clusterSize;
                    Vector n = std::accumulate(vpls.begin(), vpls.end(), Vector(0.0f), [](Vector acc, VPL arg)
                                               { return acc + arg.its.geoFrame.n; }) /
                               (float)clusterSize;
                    n = n / n.length();
                    float radius = std::accumulate(vpls.begin(), vpls.end(), 0.0f, [center](float acc, VPL arg)
                                                   { return std::max(acc, distance(arg.its.p, center)); });
                    Spectrum totalEmission = std::accumulate(vpls.begin(), vpls.end(), Spectrum(0.0f), [this](Spectrum acc, VPL arg)
                                                             { return acc + diffuseVPLEmission(arg); });
                    clusterVec[i] = VPLCluster{center + Epsilon * n, Normal(n), totalEmission, radius};
                } else {
                    clusterVec[i] = VPLCluster{Point(FLT_MAX,FLT_MAX,FLT_MAX), Normal(1.0f,0.0f,0.0f), Spectrum(0.0f), 0};
                }
            }
            refinementIteration++;
        }
        printf("VPL: %zi / %i in %i paths \n", vplVec.size(), sceneProperty_VPLCount, generatedPaths);
        float avgR = std::accumulate(clusterVec.begin(), clusterVec.end(), 0.0f, [](float acc, VPLCluster arg){return acc + arg.r;}) / (float) clusterVec.size();
        float minR = std::accumulate(clusterVec.begin(), clusterVec.end(), FLT_MAX, [](float acc, VPLCluster arg){return std::min(acc,arg.r);});
        float maxR = std::accumulate(clusterVec.begin(), clusterVec.end(), 0.0f, [](float acc, VPLCluster arg){return std::max(acc,arg.r);});
        int minClusterSize = *std::min_element(clusterSizes.begin(), clusterSizes.end());
        int maxClusterSize = *std::max_element(clusterSizes.begin(),clusterSizes.end());
        int totalVPLsClustered = std::accumulate(clusterSizes.begin(), clusterSizes.end(), 0, [](int acc, int arg){return acc+arg;});
        Vector rgbV, rgbC;
        std::accumulate(vplVec.begin(),vplVec.end(),Spectrum(0.0f),[this](Spectrum acc, VPL arg){return acc + diffuseVPLEmission(arg);}).toLinearRGB(rgbV[0],rgbV[1], rgbV[2]);
        std::accumulate(clusterVec.begin(),clusterVec.end(),Spectrum(0.0f),[](Spectrum acc, VPLCluster arg){return acc + arg.totalEmission;}).toLinearRGB(rgbC[0],rgbC[1],rgbC[2]);
        printf("Emission: (%f,%f,%f),(%f,%f,%f)\n", rgbV[0], rgbV[1], rgbV[2], rgbC[0], rgbC[1], rgbC[2]);
        printf("Clusters: %i, size: %i to %i, r: %f to %f, avg_r %f\n", (int)clusterVec.size(),
            *std::min_element(clusterSizes.begin(), clusterSizes.end()),
            *std::max_element(clusterSizes.begin(), clusterSizes.end()), minR, maxR, avgR);
        printf("Clustered %i vpls\n", totalVPLsClustered);
        std::vector<VPLCluster> preClusterVec(clusterVec);
        clusterVec.clear();
        std::copy_if(preClusterVec.begin(),preClusterVec.end(),std::back_inserter(clusterVec),[](VPLCluster arg){return abs(arg.n.length()-1.0f) < Epsilon && arg.totalEmission.max() > Epsilon;});
        printf("Removed %i invalid clusters\n", preClusterVec.size() - clusterVec.size());
        return true;
    }

    Spectrum evalVPLContribution(const RayDifferential &r, VPL vpl, RadianceQueryRecord &rRec) const
    {
        Vector toVPL = Vector(vpl.its.p - rRec.its.p);
        Vector d = toVPL / toVPL.length();
        RayDifferential toVPLrd(rRec.its.p + Epsilon * d, d, 0);
        Intersection its;
        if (dot(d, rRec.its.geoFrame.n) < Epsilon || dot(-toVPLrd.d, vpl.its.geoFrame.n) < Epsilon)
            return Spectrum(0.0f);
        rRec.scene->rayIntersect(toVPLrd, its);
        if (its.t + ShadowEpsilon >= toVPL.length())
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

    Spectrum evalClusterContribution(const RayDifferential &r, VPLCluster c, RadianceQueryRecord &rRec) const
    {
        
        Vector t = Vector(1.0f, 0.0f, 0.0f);
        Vector b = Vector(0.0f, 0.0f, 1.0f);
        if (abs(dot(c.n, t)) > 1 - Epsilon)
            t = Vector(0.0f, 1.0f, 0.0f);
        if (abs(dot(c.n, b)) > 1 - Epsilon)
            b = Vector(0.0f, 1.0f, 0.0f);
        t = t - dot(c.n, t) * c.n;
        t /= t.length();
        b = b - dot(c.n, b) * c.n + dot(t, b) * t;
        b /= b.length();
        float len = c.r * sqrt(rRec.sampler->next1D());
        //len = 0.0f;
        float theta = 2.0f * M_PI * rRec.sampler->next1D();
        Point sampleP = c.center;
        sampleP += len * (cos(theta) * t + sin(theta) * b);
        
        Vector toLight(sampleP - rRec.its.p);
        Vector d = toLight / toLight.length();
        RayDifferential rayDiff(rRec.its.p + Epsilon * d, d, 0);
        Intersection its;
        rRec.scene->rayIntersect(rayDiff, its);
        if (its.t + ShadowEpsilon >= toLight.length())
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
        VPLCluster bestC = clusterVec[0];
        float bestT = FLT_MAX;
        for (VPLCluster c : clusterVec)
        {
            float denom = dot(c.n,r.d);
            if (denom < -Epsilon)
            {
                float t = dot(Vector(c.center - r.o),c.n) / denom;
                Point isectP = r.o + t * r.d;
                if (t >= 0 && t < bestT && distance(c.center, r.o + t * r.d) < std::max(0.1f,c.r) && (!rRec.rayIntersect(r) || rRec.its.t >= t))
                {
                    bestC = c;
                    bestT = t;
                }
            }
        }
        Spectrum color = rRec.rayIntersect(r) ? rRec.its.getBSDF(r)->getDiffuseReflectance(rRec.its) / M_PI: Spectrum(0.0f);
        if (bestT < FLT_MAX)
        {
            //return bestC.totalEmission;
            Spectrum spec;
            //float rand = sin(dot(Vector(bestC.center), Vector(12.9898f, 78.233f,69.69f))) * 43758.5453f;
            //float ipart;
            //rand = abs(modf(rand,&ipart));
            //spec.fromLinearRGB(rand,rand,rand);
            spec.fromLinearRGB(0.5f + 0.5f * bestC.n[0],0.5f + 0.5f * bestC.n[1],0.5f + 0.5f * bestC.n[2]);
            return spec;
        }
        return color;
    }

    Spectrum debug_showVPLs(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        VPL best = vplVec[0];
        float bestT = FLT_MAX;
        for (VPL vpl : vplVec)
        {
            float denom = dot(vpl.its.geoFrame.n, r.d);
            if (abs(denom) > Epsilon)
            {
                float t = dot(Vector(vpl.its.p - r.o), vpl.its.geoFrame.n) / denom;
                Point isectP = r.o + t * r.d;
                if (t >= 0 && t < bestT && distance(vpl.its.p, r.o + t * r.d) < 0.05)
                {
                    best = vpl;
                    bestT = t;
                }
            }
        }
        if (bestT < FLT_MAX && rRec.its.t >= bestT)
        {
            //return bestC.totalEmission;
            std::vector<Vector> colors = std::vector<Vector>({Vector(1.0f, 1.0f, 1.0f), Vector(0.0f, 0.0f, 1.0f), Vector(1.0f, 0.0f, 0.0f)});
            Spectrum spec;
            if (best.bounce < colors.size())
            {
                Vector color = colors[best.bounce];
                spec.fromLinearRGB(color[0], color[1], color[2]);
            } else {
                spec.fromLinearRGB(0.5f, 0.5f, 0.5f);
            }
            return spec;
        }
        return Spectrum(0.0f);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        if (sceneProperty_debug_showClusters)
        {
            return debug_showClusters(r,rRec);
        }
        Spectrum total(0.0f);
        if (rRec.rayIntersect(r))
        {
            DirectSamplingRecord dRec(rRec.its);
            for (int i = 0; i < sceneProperty_directIlluminationSamples; i++)
            {
                Spectrum value = rRec.scene->sampleEmitterDirect(dRec, rRec.sampler->next2D());
                BSDFSamplingRecord bRec(rRec.its, -rRec.its.toLocal(r.d), rRec.its.toLocal(dRec.d));
                Spectrum bsdfVal = rRec.its.getBSDF(r)->eval(bRec);
                Spectrum direct = value * bsdfVal;
                total += direct / (float)sceneProperty_directIlluminationSamples;
            }
            Spectrum indirect = Spectrum(0.0f);
            if (sceneProperty_useClusters)
            {
                for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                {

                    uint32_t i = (uint32_t)(rRec.nextSample1D() * clusterVec.size());
                    i = i < clusterVec.size() ? (uint32_t)i : (uint32_t)clusterVec.size() - 1;
                    indirect += evalClusterContribution(r, clusterVec[i], rRec);
                }
                total += indirect * (float)clusterVec.size() / (float)sceneProperty_indirectIlluminationSamples;
            } else {
                for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                {

                    uint32_t i = (uint32_t)(rRec.nextSample1D() * vplVec.size());
                    i = i < vplVec.size() ? (uint32_t)i : (uint32_t)vplVec.size() - 1;
                    indirect += evalVPLContribution(r, vplVec[i], rRec);
                }
                total += indirect * (float)vplVec.size() / (float)sceneProperty_indirectIlluminationSamples;
            }
        }
        return total;
    }

private:
    ref<Random> m_random;
    bool sceneProperty_debug_showClusters, sceneProperty_useClusters;
    int sceneProperty_VPLCount, sceneProperty_directIlluminationSamples, sceneProperty_indirectIlluminationSamples, sceneProperty_maxVPLPathLength, sceneProperty_numberOfClusters, sceneProperty_refinementIterations;
    float sceneProperty_clampDistanceMin;
    std::vector<VPL> vplVec;
    std::vector<VPLCluster> clusterVec;
};
MTS_IMPLEMENT_CLASS_S(MyVPLBaselineIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MyVPLBaselineIntegrator, "My Baseline VPL Integrator");

MTS_NAMESPACE_END