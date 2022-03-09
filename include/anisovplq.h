#ifndef ANISOVPLQ
#define ANISOVPLQ

#include <mitsuba/render/scene.h>
#include <mitsuba/core/plugin.h>

MTS_NAMESPACE_BEGIN

struct VPL
{
    Intersection its;
    Vector w_o;
    Spectrum L_i;
    const BSDF *f;
    int bounce;
};

struct Disk
{
    Point center;
    Normal n;
    Spectrum totalEmission;
    float r;
    int numVPLs;
};

struct VPLCloud
{
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline float kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};

struct CurvatureData
{
    Vector dir1,dir2;
    float k1,k2;
};  

bool intersectDisk(const RayDifferential &r, Disk disk, float *t)
{
    float denom = dot(disk.n, r.d);
    if (abs(denom) > Epsilon)
    {
        float dist = dot(Vector(disk.center - r.o), disk.n) / denom;
        Point isectP = r.o + dist * r.d;
        if (dist >= 0 && distance(disk.center, r.o + dist * r.d) < disk.r)
        {
            *t = dist;
            return true;
        }
    }
    return false;
}

size_t uniformIndex(float rand, size_t length)
{
    size_t i = (size_t)(rand * length);
    return i < length ? i : length - 1;
}

Spectrum diffuseVPLEmission(const VPL &vpl)
{
    return vpl.L_i * vpl.f->getDiffuseReflectance(vpl.its) / M_PI;
}

Point3 getPointOnDisk(Disk c, Point2 rand)
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
    float theta = 2.0f * M_PI * rand.x;
    return c.center + c.r * sqrt(rand.y) * (cos(theta) * t + sin(theta) * b);
}

bool isInsideEllipse(Point center, Vector lengthDir, Vector widthDir, float length, float width, Point p)
{

    float distAlongLongAxis = ((p - center) - dot(p - center, widthDir) * widthDir).length();
    float distAlongShortAxis = ((p - center) - dot(p - center, lengthDir) * lengthDir).length();
    //float distAlongShortAxis = sqrt(pow(distance(p, isect.p), 2.0f) - pow(distAlongLongAxis, 2.0f));
    if (pow(distAlongLongAxis, 2.0f) / (length * length) + pow(distAlongShortAxis, 2.0f) / (width * width) > 1.0f)
    {
        return false;
    }
    return true;
}

bool pointIsOnDisk(Disk c, Point p)
{   
    Vector toP = p - c.center;
    Vector toPProjectedOnDisk = toP - dot(p - c.center, c.n) * c.n;
    float distToPlane = (toP - toPProjectedOnDisk).length();
    if (distToPlane > Epsilon  && toPProjectedOnDisk.length() < c.r)
        return true;
    return false;
}

float distanceOnDisk(Disk c, Ray ray)
{
    BSphere sphere(c.center,c.r);
    float near,far;
    sphere.rayIntersect(ray,near,far);
    return far-near;
}

MTS_NAMESPACE_END

#endif