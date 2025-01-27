#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "fp_data_type.h"

class sphere : public hitable
{
public:
    __device__ sphere() {}
    __device__ sphere(vec3 cen, FpDataType r, material *m) : center(cen), radius(r), mat_ptr(m){};
    __device__ virtual bool hit(const ray &r, FpDataType tmin, FpDataType tmax, hit_record &rec) const;
    vec3 center;
    FpDataType radius;
    material *mat_ptr;
};

__device__ bool sphere::hit(const ray &r, FpDataType t_min, FpDataType t_max, hit_record &rec) const
{
    vec3 oc = r.origin() - center;
    FpDataType a = fp_dot(r.direction(), r.direction());
    FpDataType b = fp_dot(oc, r.direction());
    FpDataType c = d_sub(fp_dot(oc, oc), d_mul(radius, radius));
    FpDataType discriminant = d_sub(d_mul(b, b), d_mul(a, c));
    if (discriminant > FpDataType(0.0f))
    {
        FpDataType temp = d_div(d_sub(d_neg(b), d_sqrt(discriminant)), a);
        if (temp < t_max && temp > t_min)
        {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + d_sqrt(discriminant)) / a;
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
