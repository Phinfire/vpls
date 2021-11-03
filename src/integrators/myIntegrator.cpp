#include <mitsuba/render/scene.h>
MTS_NAMESPACE_BEGIN
class MyIntegrator : public SamplingIntegrator
{
public:
    MTS_DECLARE_CLASS()
    /// Initialize the integrator with the specified properties
    MyIntegrator(const Properties &props) : SamplingIntegrator(props)
    {
        Spectrum defaultColor;
        defaultColor.fromLinearRGB(0.2f, 0.5f, 0.2f);
        m_color = props.getSpectrum("color", defaultColor);
    }

    /// Unserialize from a binary data stream
    MyIntegrator(Stream *stream, InstanceManager *manager)
        : SamplingIntegrator(stream, manager)
    {
        m_color = Spectrum(stream);
    }
    /// Serialize to a binary data stream
    void serialize(Stream *stream, InstanceManager *manager) const
    {
        SamplingIntegrator::serialize(stream, manager);
        m_color.serialize(stream);
    }

    /// Query for an unbiased estimate of the radiance along <tt>r</tt>
    Spectrum Li(const RayDifferential &r, RadianceQueryRecord &rRec) const
    {
        int m_directIlluminationSamples = 16;
        Spectrum total(0.0f);
        if (rRec.rayIntersect(r))
        {
            DirectSamplingRecord dRec(rRec.its);
            for (int i = 0; i < m_directIlluminationSamples; i++)
            {
                /*
                Spectrum value = rRec.scene->sampleEmitterDirect(dRec, rRec.sampler->next2D());
                BSDFSamplingRecord bRec(rRec.its, -rRec.its.toLocal(r.d), rRec.its.toLocal(dRec.d));
                Spectrum bsdfVal = rRec.its.getBSDF(r)->eval(bRec); // pdf?
                Spectrum direct = value * bsdfVal;
                total += direct / (float)m_directIlluminationSamples;
                */
                PositionSamplingRecord pRec;
                Spectrum pWeight = rRec.scene->sampleEmitterPosition(pRec, rRec.sampler->next2D());
                const Emitter *emitter = static_cast<const Emitter *>(pRec.object);
                DirectionSamplingRecord dRec;
                Spectrum dWeight = emitter->sampleDirection(dRec, pRec, rRec.sampler->next2D());
            }
        }
        return total;
    }

private:
    Spectrum m_color;
};
MTS_IMPLEMENT_CLASS_S(MyIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(MyIntegrator, "A contrived integrator");
MTS_NAMESPACE_END