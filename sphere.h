#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "fp_data_type.h"

class sphere : public hitable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m){};
    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;
    vec3 center;
    float radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    FpDataType a = fp_dot(r.direction(), r.direction());
    FpDataType b = fp_dot(oc, r.direction());
    FpDataType c = __hsub(fp_dot(oc, oc),
                          __hmul(__float2half(radius), __float2half(radius)));
    float discriminant = __half2float(__hsub(__hmul(b, b), __hmul(a, c)));
    if (discriminant > 0.0f)
    {
        float temp = (-__half2float(b) - sqrt(discriminant)) / __half2float(a);
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + hsqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
    }
    return false;
}

#endif
