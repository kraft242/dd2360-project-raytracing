#ifndef FP_DATA_TYPE_H
#define FP_DATA_TYPE_H

#include <cuda_fp16.h>

typedef __half FpDataType; // This is the data type that will be used for all floating point operations

#define DATA_TYPE_MAX 65504.0 // The maximum value that the data type can hold

//define some constants
#define FP_DATA_TYPE_ZERO FpDataType(0.0f)
#define FP_DATA_TYPE_ONE FpDataType(1.0f)
#define FP_DATA_TYPE_TWO FpDataType(2.0f)
#define FP_DATA_TYPE_THREE FpDataType(3.0f)
#define FP_DATA_TYPE_FOUR FpDataType(4.0f)
#define FP_DATA_TYPE_FIVE FpDataType(5.0f)

//Gives the datatype a way to calculate the absolute value
__device__ inline FpDataType habs(const __half a) {
    __half_raw abs_a_raw = static_cast<__half_raw>(a);
    abs_a_raw.x &= (unsigned short)0x7FFFU;
    if (abs_a_raw.x > (unsigned short)0x7C00U)
    {
        // return canonical NaN
        abs_a_raw.x = (unsigned short)0x7FFFU;
    }
    return static_cast<__half>(abs_a_raw);
}

//Gives the datatype a way to calculate the power of a number
//More exactly, this function takes in a base and the exponent and gives back base^exponent
__device__ FpDataType hpow(const FpDataType base, const FpDataType power)
{
    return hexp(power * hlog(base));
}

//Gives the datatype a way to calculate the tangent of an angle
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

//Gives the datatype a way to calculate the square root of a value (v)
__device__ inline FpDataType d_sqrt(const FpDataType v) {
    return hsqrt(v);
}

//Gives the datatype a way to calculate the multiplication of two values
__device__ inline FpDataType d_mul(const FpDataType a, const FpDataType b) {
    return a * b;
}

//Gives the datatype a way to calculate the division of two values
__device__ inline FpDataType d_div(const FpDataType a, const FpDataType b) {
    return a / b;
}

//Gives the datatype a way to calculate the addition of two values
__device__ inline FpDataType d_add(const FpDataType a, const FpDataType b) {
    return a + b;
}

//Gives the datatype a way to calculate the subtraction of two values
__device__ inline FpDataType d_sub(const FpDataType a, const FpDataType b) {
    return a - b;
}

//Gives the datatype a way to calculate the negation of a value
__device__ inline FpDataType d_neg(const FpDataType a) {
    return a * FpDataType(-1);
}

//A different name for calculating the tangent of an angle
__device__ inline FpDataType d_tan(const FpDataType a) {
    return htan(a);
}

//A different name for calculating the base*exponent
__device__ inline FpDataType d_pow(const FpDataType a, const FpDataType b) {
    return hpow(a, b);
}

//The conversion of a float to the data type
__device__ inline FpDataType floatToDataType(const float v) {
    return FpDataType(v);
}

#endif // FP_DATA_TYPE_H