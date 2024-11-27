#ifndef CAMERAH
#define CAMERAH

#include <curand_kernel.h>
#include "ray.h"
#include "fp_data_type.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

__device__ vec3 random_in_unit_disk(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = FP_DATA_TYPE_TWO * vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), 0) - vec3(1, 1, 0);
    } while (dot(p, p) >= FP_DATA_TYPE_ONE);
    return p;
}

class camera
{
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, FpDataType vfov, FpDataType aspect, FpDataType aperture, FpDataType focus_dist)
    { // vfov is top to bottom in degrees
        lens_radius = aperture / FP_DATA_TYPE_TWO;
        FpDataType theta = vfov * ((FpDataType)M_PI) / FpDataType(180.0f);
        FpDataType half_height = htan(theta / FP_DATA_TYPE_TWO);
        FpDataType half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = FP_DATA_TYPE_TWO * half_width * focus_dist * u;
        vertical = FP_DATA_TYPE_TWO * half_height * focus_dist * v;
    }
    __device__ ray get_ray(FpDataType s, FpDataType t, curandState *local_rand_state)
    {
        vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
        vec3 offset = u * rd.x() + v * rd.y();
        return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    vec3 origin;
    vec3 lower_left_corner;
    vec3 horizontal;
    vec3 vertical;
    vec3 u, v, w;
    FpDataType lens_radius;
};

#endif
