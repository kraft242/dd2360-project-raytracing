#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "fp_data_type.h"

class vec3
{

public:
    __device__ vec3() {}
    __device__ vec3(FpDataType e0, FpDataType e1, FpDataType e2)
    {
        e[0] = e0;
        e[1] = e1;
        e[2] = e2;
    }
    __host__ __device__ inline FpDataType x() const { return e[0]; }
    __host__ __device__ inline FpDataType y() const { return e[1]; }
    __host__ __device__ inline FpDataType z() const { return e[2]; }
    __host__ __device__ inline FpDataType r() const { return e[0]; }
    __host__ __device__ inline FpDataType g() const { return e[1]; }
    __host__ __device__ inline FpDataType b() const { return e[2]; }

    __device__ inline const vec3 &operator+() const { return *this; }
    __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __device__ inline FpDataType operator[](int i) const { return e[i]; }
    __device__ inline FpDataType &operator[](int i) { return e[i]; };

    __device__ inline vec3 &operator+=(const vec3 &v2);
    __device__ inline vec3 &operator-=(const vec3 &v2);
    __device__ inline vec3 &operator*=(const vec3 &v2);
    __device__ inline vec3 &operator/=(const vec3 &v2);
    __device__ inline vec3 &operator*=(const FpDataType t);
    __device__ inline vec3 &operator/=(const FpDataType t);

    __device__ inline FpDataType length() const { return d_sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __device__ inline FpDataType squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __device__ inline void make_unit_vector();

    FpDataType e[3];
};

inline std::istream &operator>>(std::istream &is, vec3 &t)
{
    float v0 = 0;
    float v1 = 0;
    float v2 = 0;
    is >> v0 >> v1 >> v2;
    t.e[0] = float(v0);
    t.e[1] = float(v1);
    t.e[2] = float(v2);
    return is;
}

inline std::ostream &operator<<(std::ostream &os, const vec3 &t)
{
    os << float(t.e[0]) << " " << float(t.e[1]) << " " << float(t.e[2]);
    return os;
}

__device__ inline void vec3::make_unit_vector()
{
    FpDataType k = FpDataType(1.0f) / d_sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
}

__device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2)
{
    return vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__device__ inline vec3 operator*(FpDataType t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ inline vec3 operator/(vec3 v, FpDataType t)
{
    return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__device__ inline vec3 operator*(const vec3 &v, FpDataType t)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__device__ inline FpDataType dot(const vec3 &v1, const vec3 &v2)
{
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__device__ inline FpDataType d_dot(const vec3 &a, const vec3 &b) {
    return a.e[0] * b.e[0] + a.e[1] * b.e[1] + a.e[2] * b.e[2];
}

__device__ inline FpDataType fp_dot(const vec3 &v1, const vec3 &v2)
{
    return d_add((v1.x()) * (v2.x()),
                  d_add((v1.y()) * (v2.y()),
                         (v1.z()) * (v2.z())));
}

            
__device__ inline vec3 cross(const vec3 &v1, const vec3 &v2)
{
    return vec3((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
                (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
                (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}

__device__ inline vec3 &vec3::operator+=(const vec3 &v)
{
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator*=(const vec3 &v)
{
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator/=(const vec3 &v)
{
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator-=(const vec3 &v)
{
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__device__ inline vec3 &vec3::operator*=(const FpDataType t)
{
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__device__ inline vec3 &vec3::operator/=(const FpDataType t)
{
    FpDataType k = FpDataType(1.0) / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__device__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

#endif
