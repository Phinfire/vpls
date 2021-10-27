#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/hw/vpl.h>

#include <vector>
#include <algorithm>

MTS_NAMESPACE_BEGIN

class myVPLBaselineIntegrator : public SamplingIntegrator
{
public:
    MTS_DECLARE_CLASS()

    myVPLBaselineIntegrator(const Properties &props) : SamplingIntegrator(props)
    {
        m_directIlluminationSamples = props.getInteger("directIlluminationSamples", 0);
        desiredVPLCount = props.getInteger("desiredVPLCount", 0);
        m_maxVPLPathlength = props.getInteger("maxPathLength", 1);
        clampDistanceTo = props.getFloat("clampDistanceTo", 1.0f);
    }

    myVPLBaselineIntegrator(Stream *stream, InstanceManager *manager) : SamplingIntegrator(stream, manager)
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
            Properties props("halton");
            props.setInteger("scramble", 0);
            sampler = static_cast<Sampler *>(PluginManager::getInstance()->createObject(MTS_CLASS(Sampler), props));
            sampler->configure();
            sampler->generate(Point2i());
        }
        size_t offset = 5;
        int generatedPaths = 0;
        for (int i = 0; i < 1000000 && vplVec.size() < desiredVPLCount; i++)
        {
            float rr_weight = 1;
            sampler->setSampleIndex(++offset);
            PositionSamplingRecord pRecOnEmitter;
            Spectrum pWeight = scene->sampleEmitterPosition(pRecOnEmitter, sampler->next2D());
            const Emitter *emitter = static_cast<const Emitter *>(pRecOnEmitter.object);
            DirectionSamplingRecord dRec;
            Spectrum dWeight = emitter->sampleDirection(dRec, pRecOnEmitter, sampler->next2D());
            Intersection isectOnEmitter;
            RayDifferential flipRay(pRecOnEmitter.p + dRec.d, -dRec.d,0);
            scene->rayIntersect(flipRay, isectOnEmitter);
            Intersection its;
            RayDifferential ray(pRecOnEmitter.p, dRec.d, 0);
            float albedo = 1;
            int pathDepth = 0;
            Spectrum currentPathFlux = pWeight * dWeight * dot(isectOnEmitter.geoFrame.n, dRec.d);
            while (sampler->next1D() > 1 - albedo && scene->rayIntersect(ray, its) && vplVec.size() < desiredVPLCount && pathDepth < m_maxVPLPathlength)
            {
                if (pathDepth == 0)
                    generatedPaths++;
                myVPL v;
                v.L_i = currentPathFlux;
                v.w_o = -ray.d;
                v.its = its;
                v.f = its.getBSDF(ray);
                vplVec.push_back(v);
                BSDFSamplingRecord nextBRec(its, sampler, EImportance);
                Spectrum cBsdfVal = its.getBSDF()->sample(nextBRec, sampler->next2D());
                BSDFSamplingRecord cBsdfRec(its, -ray.d, its.toWorld(nextBRec.wo));
                ray = Ray(its.p, its.toWorld(nextBRec.wo), 0.0f);
                currentPathFlux = cBsdfVal * std::max(0.0f, dot(its.geoFrame.n, its.toWorld(nextBRec.wo))) * currentPathFlux / albedo;
                albedo = std::min(0.95f, cBsdfVal.max());
                pathDepth++;
                if (cBsdfVal.isZero())
                {
                    printf("vpl %zi emission was zero \n", vplVec.size());
                    break;
                }
            }
        }
        for (int i = 0; i < vplVec.size(); i++)
        {
            vplVec[i].L_i /= (float)generatedPaths;
        }
        printf("VLPs generated: %zi / %i in %i paths\n", vplVec.size(), desiredVPLCount, generatedPaths);
        return true;
    }

    void printSpec(const Spectrum &spec)
    {
        Float r, g, b;
        spec.toLinearRGB(r, g, b);
        printf("(%f,%f,%f)\n", r, g, b);
    }

    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        Spectrum total(0.0f);
        if (rRec.rayIntersect(r))
        {
            DirectSamplingRecord dRec(rRec.its);
            for (int i = 0; i < m_directIlluminationSamples; i++)
            {
                Spectrum value = rRec.scene->sampleEmitterDirect(dRec, rRec.sampler->next2D());
                BSDFSamplingRecord bRec(rRec.its, -rRec.its.toLocal(r.d), rRec.its.toLocal(dRec.d));
                Spectrum bsdfVal = rRec.its.getBSDF(r)->eval(bRec);
                Spectrum direct = value * bsdfVal;
                total += direct / (float)m_directIlluminationSamples;
            }
            float minDist = 1000000;
            for (int i = 0; i < vplVec.size(); i++)
            {
                Vector toVPL = Vector(vplVec[i].its.p - rRec.its.p);
                minDist = minDist < toVPL.length() ? minDist : toVPL.length();
            }
            return Spectrum(0.01f / minDist);
            //
            Spectrum indirect = Spectrum(0.0f);
            float epsilonScale = 1;
            for (int i = 0; i < vplVec.size(); i++)
            {
                Vector toVPL = Vector(vplVec[i].its.p - rRec.its.p);
                Vector d = toVPL / toVPL.length();
                RayDifferential toVPLrd(rRec.its.p + Epsilon * d, d, 0);
                Intersection its;
                if (dot(d, rRec.its.geoFrame.n) < epsilonScale * Epsilon || dot(-toVPLrd.d, vplVec[i].its.geoFrame.n) < epsilonScale * Epsilon)
                {
                    continue;
                }
                rRec.scene->rayIntersect(toVPLrd, its);
                if (its.t + ShadowEpsilon >= toVPL.length()) // vpl is visible
                {
                    BSDFSamplingRecord bRec_x2(vplVec[i].its, vplVec[i].its.toLocal(-toVPLrd.d), vplVec[i].its.toLocal(vplVec[i].w_o));
                    BSDFSamplingRecord bRec_x1(rRec.its, rRec.its.toLocal(-r.d), rRec.its.toLocal(toVPLrd.d));
                    Spectrum f_x2 = vplVec[i].f->eval(bRec_x2);
                    Spectrum f_x1 = rRec.its.getBSDF(r)->eval(bRec_x1);
                    Float geo_yx = std::max(dot(-d, Vector(vplVec[i].its.geoFrame.n)), 0.0f);
                    Float clampedDistance = std::max(toVPL.length() * toVPL.length(), clampDistanceTo * clampDistanceTo);
                    //clampedDistance = toVPL.length() * toVPL.length();
                    Float geometryTerm = geo_yx / clampedDistance; // geo_xy is already evaluated in f->eval
                    indirect += f_x1 * geometryTerm * f_x2 * vplVec[i].L_i;
                }
            }
            total += indirect;
        }
        return total;
    }

private:
    ref<Random> m_random;
    int desiredVPLCount;
    int m_directIlluminationSamples;
    int m_maxVPLPathlength;
    float clampDistanceTo;
    struct myVPL
    {
        Intersection its;
        Vector w_o;
        Spectrum L_i;
        const BSDF *f;
    };
    std::vector<myVPL> vplVec;
};
MTS_IMPLEMENT_CLASS_S(myVPLBaselineIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(myVPLBaselineIntegrator, "My Baseline VPL Integrator");

MTS_NAMESPACE_END