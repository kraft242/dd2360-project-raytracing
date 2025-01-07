#ifndef FP_DATA_TYPE_H
#define FP_DATA_TYPE_H

#include <cuda_bf16.h> // include the bfloat from cuda

typedef nv_bfloat16 FpDataType; // define the data type

// define some constants
#define FP_DATA_TYPE_ZERO FpDataType(0.0f)
#define FP_DATA_TYPE_ONE FpDataType(1.0f)
#define FP_DATA_TYPE_TWO FpDataType(2.0f)
#define FP_DATA_TYPE_THREE FpDataType(3.0f)
#define FP_DATA_TYPE_FOUR FpDataType(4.0f)
#define FP_DATA_TYPE_FIVE FpDataType(5.0f)

//Gived the data type a way to calculate the absolute value of a value
__device__ inline FpDataType habs(const nv_bfloat16 a) {
    return __habs(a);
}

// Give the data type a power of function
//This will take in a base and the exponent and give back base^exponent
__device__ FpDataType hpow(const FpDataType base, const FpDataType power)
{
    return hexp(power * hlog(base));
}

// Give the data type a way to calculate the tangent of an angle
__device__ FpDataType htan(const FpDataType angle)
{
    const FpDataType cos = hcos(angle);
    const FpDataType sin = hsin(angle);
    const FpDataType tolerance = FpDataType(1e-3);
    if (habs(cos) < tolerance)
    {
        return FP_DATA_TYPE_ZERO;
    }
    return sin / cos;
}

// Give the data type a way to calculate the square root of a value (v)
__device__ inline FpDataType d_sqrt(const FpDataType v) {
    return hsqrt(v);
}

// Give the data type a way to calculate the multiplication of two values
__device__ inline FpDataType d_mul(const FpDataType a, const FpDataType b) {
    return a * b;
}

// Give the data type a way to calculate the division of two values
__device__ inline FpDataType d_div(const FpDataType a, const FpDataType b) {
    return a / b;
}

// Give the data type a way to calculate the addition of two values
__device__ inline FpDataType d_add(const FpDataType a, const FpDataType b) {
    return a + b;
}

// Give the data type a way to calculate the subtraction of two values
__device__ inline FpDataType d_sub(const FpDataType a, const FpDataType b) {
    return a - b;
}

// Give the data type a way to calculate the negation of a value
__device__ inline FpDataType d_neg(const FpDataType a) {
    return a * FpDataType(-1);
}

// Give the data type a way to calculate the tangent of an angle
__device__ inline FpDataType d_tan(const FpDataType a) {
    return htan(a);
}

//A different name to call the powe function to calculate the base*exponent
__device__ inline FpDataType d_pow(const FpDataType a, const FpDataType b) {
    return hpow(a, b);
}

// The conversion of a float to the data type
__device__ inline FpDataType floatToDataType(const float v) {
    return FpDataType(v);
}

#endif // FP_DATA_TYPE_H