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
    };

    MyVPLBaselineIntegrator(const Properties &props) : SamplingIntegrator(props)
    {
        sceneProperty_directIlluminationSamples = props.getInteger("directIlluminationSamples", 0);
        sceneProperty_indirectIlluminationSamples = props.getInteger("indirectIlluminationSamples", 0);
        sceneProperty_numVPLSamples = props.getInteger("indirectIlluminationSamples", 0);
        sceneProperty_VPLCount = props.getInteger("desiredVPLCount", 0);
        sceneProperty_maxVPLPathLength = props.getInteger("maxPathLength", 1);
        sceneProperty_clampDistanceMin = props.getFloat("clampDistanceTo", 1.0f);
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
            if (sceneProperty_indirectIlluminationSamples < vplVec.size())
            {
                for (int x_ = 0; x_ < sceneProperty_indirectIlluminationSamples; x_++)
                {

                    uint32_t i = (uint32_t)(rRec.nextSample1D() * vplVec.size());
                    i = i < vplVec.size() ? i : vplVec.size() - 1;
                    indirect += evalVPLContribution(r, vplVec[i], rRec);
                }
                total += indirect * (float)vplVec.size() / (float)sceneProperty_indirectIlluminationSamples;
            } else {
                for (MyVPL vpl : vplVec) {
                    indirect += evalVPLContribution(r, vpl, rRec);
                }
            }
        }
        return total;
    }

private:
    ref<Random> m_random;
    int sceneProperty_VPLCount, sceneProperty_numVPLSamples, sceneProperty_directIlluminationSamples, sceneProperty_indirectIlluminationSamples, sceneProperty_maxVPLPathLength;
    float sceneProperty_clampDistanceMin;
    std::vector<MyVPL> vplVec;
    std::vector<VPLCluster> clusterVec;
};
MTS_IMPLEMENT_CLASS_S(MyVPLBaselineIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MyVPLBaselineIntegrator, "My Baseline VPL Integrator");

MTS_NAMESPACE_END