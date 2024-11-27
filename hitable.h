#ifndef HITABLEH
#define HITABLEH

#include "ray.h"
#include "fp_data_type.h"

class material;

struct hit_record
{
    FpDataType t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, FpDataType t_min, FpDataType t_max, hit_record& rec) const = 0;
};

#endif
