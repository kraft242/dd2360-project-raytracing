#ifndef SPHEREH
#define SPHEREH

#include "hitable.h"
#include "fp_data_type.h"

class sphere: public hitable  {
    public:
        __device__ sphere() {}
        __device__ sphere(vec3 cen, FpDataType r, material *m) : center(cen), radius(r), mat_ptr(m)  {};
        __device__ virtual bool hit(const ray& r, FpDataType tmin, FpDataType tmax, hit_record& rec) const;
        vec3 center;
        FpDataType radius;
        material *mat_ptr;
};

__device__ bool sphere::hit(const ray& r, FpDataType t_min, FpDataType t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    FpDataType a = dot(r.direction(), r.direction());
    FpDataType b = dot(oc, r.direction());
    FpDataType c = dot(oc, oc) - radius*radius;
    FpDataType discriminant = b*b - a*c;
    if (discriminant > 0) {
        FpDataType temp = (-b - sqrt(discriminant))/a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - center) / radius;
            rec.mat_ptr = mat_ptr;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
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
