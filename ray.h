#ifndef RAYH
#define RAYH
#include "vec3.h"
#include "fp_data_type.h"

class ray
{
    public:
        __device__ ray() {}
        __device__ ray(const vec3& a, const vec3& b) { A = a; B = b; }
        __device__ vec3 origin() const       { return A; }
        __device__ vec3 direction() const    { return B; }
        __device__ vec3 point_at_parameter(FpDataType t) const { return A + t*B; }

        vec3 A;
        vec3 B;
};

#endif
