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

    struct MyVPL
    {
        Intersection its;
        Vector w_o;
        Spectrum L_i;
        const BSDF *f;
    };

    struct VPLCluster
    {
        Point center;
        Normal n;
        Spectrum flux;
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
            //float rr_weight = 1;
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
                MyVPL v;
                v.L_i = currentPathFlux;
                v.w_o = -ray.d;
                v.its = its;
                v.f = its.getBSDF(ray);
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
        clusterVec.clear();
        int numClusters = 10;
        float w_dist = 1, w_n = 1, w_flux = 1;
        std::function<float(MyVPL, MyVPL)> distanceMetric = [w_dist, w_n, w_flux](const MyVPL &vpl1, const MyVPL &vpl2)
        {
            Vector rgb1, rgb2;
            vpl1.L_i.toLinearRGB(rgb1[0], rgb1[1], rgb1[2]);
            vpl2.L_i.toLinearRGB(rgb2[0], rgb2[1], rgb2[2]);
            rgb1 /= rgb1.length();
            rgb2 /= rgb2.length();
            return w_dist * distance(vpl1.its.p, vpl2.its.p) + w_n * (1 - dot(vpl1.its.geoFrame.n, vpl2.its.geoFrame.n)) + w_flux * dot(rgb2 - rgb1, rgb2 - rgb1);
        };
        std::function<float(VPLCluster, MyVPL)> clusterAndVPLdistanceMetric = [w_dist, w_n, w_flux](const VPLCluster &cluster, const MyVPL &vpl)
        {
            Vector rgb1, rgb2;
            cluster.flux.toLinearRGB(rgb1[0], rgb1[1], rgb1[2]);
            vpl.L_i.toLinearRGB(rgb2[0], rgb2[1], rgb2[2]);
            rgb1 /= rgb1.length();
            rgb2 /= rgb2.length();
            return w_dist * distance(cluster.center, vpl.its.p) + w_n * (1 - dot(cluster.n, vpl.its.geoFrame.n)) + w_flux * dot(rgb2 - rgb1, rgb2 - rgb1);
        };
        std::set<int> clusterInits;
        for (int i = 0; i < sceneProperty_numberOfClusters; i++)
        {
            while (clusterInits.size() <= i)
            {
                int idx = (int)(sampler->next1D() * vplVec.size());
                clusterInits.insert(std::min(idx, (int)vplVec.size() - 1));
            }
        }
        std::transform(clusterInits.begin(),clusterInits.end(),std::back_inserter(clusterVec), [this](const int arg){ VPLCluster c = {vplVec[arg].its.p,vplVec[arg].its.geoFrame.n,vplVec[arg].L_i,0.0f}; return c; });
        std::vector<int> clusterAssignmentsPrev;
        std::vector<int> clusterAssignments(vplVec.size(),-1);
        int refinementIteration = 0;
        do {
            printf("refining (%i)...\n", refinementIteration);
            clusterAssignmentsPrev = clusterAssignments;
            for (int i = 0; i < vplVec.size(); i++)
            {
                std::vector<VPLCluster>::iterator minCluster = min_element(clusterVec.begin(), clusterVec.end(), [this, i, clusterAndVPLdistanceMetric](VPLCluster arg1, VPLCluster arg2)
                {
                    return clusterAndVPLdistanceMetric(arg1, vplVec[i]) < clusterAndVPLdistanceMetric(arg2, vplVec[i]);
                });
                clusterAssignments[i] = (int) std::distance(clusterVec.begin(), minCluster);
            }
            for (int i = 0; i < clusterVec.size(); i++)
            {
                Point center_acc = Point(0.0f);
                Vector n_acc = Vector(0.0f);
                Spectrum flux_acc = Spectrum(0.0f);
                int counter = 0;
                for (int j = 0; j < vplVec.size(); j++)
                {
                    if (clusterAssignments[j] == i)
                    {
                        n_acc += vplVec[j].its.geoFrame.n;
                        counter++;
                    }
                }
                Vector n = n_acc / (float)counter;
                for (int j = 0; j < vplVec.size(); j++)
                {
                    if (clusterAssignments[j] == i)
                    {
                        center_acc += vplVec[j].its.p;
                        n_acc += vplVec[j].its.geoFrame.n;
                        BSDFSamplingRecord bsdfRec(vplVec[j].its, vplVec[j].its.toLocal(n / n.length()), vplVec[j].its.toLocal(vplVec[j].w_o));
                        Spectrum brdfVal = vplVec[j].f->eval(bsdfRec);
                        flux_acc += vplVec[j].L_i * brdfVal * std::max(0.0f, dot(n / n.length(), vplVec[j].its.geoFrame.n));
                    }
                }
                float r = 0;
                Point center = center_acc / (float)counter;
                for (int j = 0; j < vplVec.size(); j++)
                {
                    if (clusterAssignments[j] == i)
                    {
                        r = std::max(r,distance(vplVec[j].its.p, center));
                    }
                }
                clusterVec[i] = VPLCluster{center, Normal(n / n.length()), flux_acc,r};
            }
            refinementIteration++;
        } while ((refinementIteration < 5 && !equal(clusterAssignmentsPrev.begin(), clusterAssignmentsPrev.end(), clusterAssignments.begin())));
        printf("VPL generated: %zi / %i in %i paths || %i clusters\n", vplVec.size(), sceneProperty_VPLCount, generatedPaths, (int) clusterVec.size());
        return true;
    }

    Spectrum evalVPLContribution(const RayDifferential &r, MyVPL vpl, RadianceQueryRecord &rRec) const
    {
        Vector toVPL = Vector(vpl.its.p - rRec.its.p);
        Vector d = toVPL / toVPL.length();
        RayDifferential toVPLrd(rRec.its.p + Epsilon * d, d, 0);
        Intersection its;
        if (dot(d, rRec.its.geoFrame.n) < Epsilon || dot(-toVPLrd.d, vpl.its.geoFrame.n) < Epsilon)
        {
            return Spectrum(0.0f);
        }
        rRec.scene->rayIntersect(toVPLrd, its);
        if (its.t + ShadowEpsilon >= toVPL.length())
        {
            BSDFSamplingRecord bRec_x2(vpl.its, vpl.its.toLocal(-toVPLrd.d), vpl.its.toLocal(vpl.w_o));
            BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(toVPLrd.d));
            Spectrum f_x2 = vpl.f->eval(bRec_x2);
            Spectrum f_x1 = rRec.its.getBSDF(r)->eval(bRec_x1);
            Float geo_yx = std::max(dot(-d, Vector(vpl.its.geoFrame.n)), 0.0f);
            Float clampedDistance = std::max(toVPL.length(), sceneProperty_clampDistanceMin);
            Float geometryTerm = geo_yx / pow(clampedDistance, 2.0f);
            return f_x1 * geometryTerm * f_x2 / bRec_x2.wo.z * vpl.L_i;
        }
        return Spectrum(0.0f);
    }

    Spectrum evalClusterContribution(const RayDifferential &r, VPLCluster c, RadianceQueryRecord &rRec) const
    {
        Vector toCenter = Vector(c.center - rRec.its.p);
        float cos_x = std::max(0.0f, dot(toCenter / (float)toCenter.length(), rRec.its.geoFrame.n));
        float cos_xc = std::max(0.0f, dot(-toCenter / (float)toCenter.length(), c.n));
        Intersection its;
        rRec.scene->rayIntersect(RayDifferential(rRec.its.p + Epsilon * toCenter / toCenter.length(), toCenter / toCenter.length(), 0), its);
        if (cos_x > Epsilon && cos_xc > Epsilon && its.t + ShadowEpsilon >= toCenter.length())
        {
            float diskArea = M_PI * c.r * c.r;
            BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(toCenter / toCenter.length()));
            Spectrum L_i = c.flux * diskArea * cos_x * cos_xc / (diskArea + M_PI * pow(toCenter.length(), 2.0f));
            return L_i * rRec.its.getBSDF(r)->eval(bRec_x1);
        }
        return Spectrum(0.0f);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
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
            if (true)
            {
                for (VPLCluster c : clusterVec)
                {
                    total += evalClusterContribution(r, c, rRec);
                }
            } else {
                if (sceneProperty_indirectIlluminationSamples < vplVec.size())
                {
                    for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                    {

                        uint32_t i = (uint32_t)(rRec.nextSample1D() * vplVec.size());
                        i = i < vplVec.size() ? (uint32_t)i : (uint32_t)vplVec.size() - 1;
                        indirect += evalVPLContribution(r, vplVec[i], rRec);
                    }
                    total += indirect * (float)vplVec.size() / (float)sceneProperty_indirectIlluminationSamples;
                }
                else
                {
                    for (MyVPL vpl : vplVec)
                    {
                        indirect += evalVPLContribution(r, vpl, rRec);
                    }
                }
            }
        }
        return total;
    }

private:
    ref<Random> m_random;
    int sceneProperty_VPLCount, sceneProperty_directIlluminationSamples, sceneProperty_indirectIlluminationSamples, sceneProperty_maxVPLPathLength, sceneProperty_numberOfClusters;
    float sceneProperty_clampDistanceMin;
    std::vector<MyVPL> vplVec;
    std::vector<VPLCluster> clusterVec;
};
MTS_IMPLEMENT_CLASS_S(MyVPLBaselineIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MyVPLBaselineIntegrator, "My Baseline VPL Integrator");

MTS_NAMESPACE_END