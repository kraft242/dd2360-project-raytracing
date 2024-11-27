#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"
#include "fp_data_type.h"


__device__ FpDataType schlick(FpDataType cosine, FpDataType ref_idx)
{
    FpDataType r0 = (FP_DATA_TYPE_ONE - ref_idx) / (FP_DATA_TYPE_ZERO + ref_idx);
    r0 = r0 * r0;
    return r0 + (FP_DATA_TYPE_ONE - r0) * hpow((FP_DATA_TYPE_ONE - cosine), FpDataType(5.0f));
}

__device__ bool refract(const vec3 &v, const vec3 &n, FpDataType ni_over_nt, vec3 &refracted)
{
    vec3 uv = unit_vector(v);
    FpDataType dt = dot(uv, n);
    FpDataType discriminant = FP_DATA_TYPE_ONE - ni_over_nt * ni_over_nt * (FP_DATA_TYPE_ONE - dt * dt);
    if (discriminant > FP_DATA_TYPE_ZERO)
    {
        refracted = ni_over_nt * (uv - n * dt) - n * hsqrt(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state)
{
    vec3 p;
    do
    {
        p = FP_DATA_TYPE_TWO * RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= FP_DATA_TYPE_ONE);
    return p;
}

__device__ vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - FP_DATA_TYPE_TWO * dot(v, n) * n;
}

class material
{
public:
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material
{
public:
    __device__ lambertian(const vec3 &a) : albedo(a) {}
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
        scattered = ray(rec.p, target - rec.p);
        attenuation = albedo;
        return true;
    }

    vec3 albedo;
};

class metal : public material
{
public:
    __device__ metal(const vec3 &a, FpDataType f) : albedo(a)
    {
        if (f < FP_DATA_TYPE_ONE)
            fuzz = f;
        else
            fuzz = 1;
    }
    __device__ virtual bool scatter(const ray &r_in, const hit_record &rec, vec3 &attenuation, ray &scattered, curandState *local_rand_state) const
    {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(local_rand_state));
        attenuation = albedo;
        return (dot(scattered.direction(), rec.normal) > FP_DATA_TYPE_ZERO);
    }
    vec3 albedo;
    FpDataType fuzz;
};

class dielectric : public material
{
public:
    __device__ dielectric(FpDataType ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray &r_in,
                                    const hit_record &rec,
                                    vec3 &attenuation,
                                    ray &scattered,
                                    curandState *local_rand_state) const
    {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        FpDataType ni_over_nt;
        attenuation = vec3(1.0, 1.0, 1.0);
        vec3 refracted;
        FpDataType reflect_prob;
        FpDataType cosine;
        if (dot(r_in.direction(), rec.normal) > FP_DATA_TYPE_ZERO)
        {
            outward_normal = -rec.normal;
            //outward_normal = vec3(-rec.normal.x(), -rec.normal.y(), -rec.normal.z());
            ni_over_nt = ref_idx;
            cosine = dot(r_in.direction(), rec.normal) / r_in.direction().length();
            cosine = hsqrt(FP_DATA_TYPE_ONE - ref_idx * ref_idx * (FP_DATA_TYPE_ONE - cosine * cosine));
        }
        else
        {
            outward_normal = rec.normal;
            ni_over_nt = FP_DATA_TYPE_ONE / ref_idx;
            cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = FP_DATA_TYPE_ONE;
        if (FpDataType(curand_uniform(local_rand_state)) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    FpDataType ref_idx;
};
#endif
