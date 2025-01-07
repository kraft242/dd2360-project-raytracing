#ifndef FP_DATA_TYPE_H
#define FP_DATA_TYPE_H

typedef double FpDataType; // Define what the floating point data type is

// Define some constants
#define FP_DATA_TYPE_ZERO FpDataType(0.0f)
#define FP_DATA_TYPE_ONE FpDataType(1.0f)
#define FP_DATA_TYPE_TWO FpDataType(2.0f)
#define FP_DATA_TYPE_THREE FpDataType(3.0f)
#define FP_DATA_TYPE_FOUR FpDataType(4.0f)
#define FP_DATA_TYPE_FIVE FpDataType(5.0f)

// Give the data type a absolut function
__device__ inline FpDataType habs(const FpDataType a) {
  return std::abs(a);
}

// Give the data type a power of function
// This function takes in a base and the exponent and gives back base^exponent
__device__ FpDataType hpow(const FpDataType base, const FpDataType power)
{
    return std::pow(base, power);
}

// Give the data type a way to calculate the tangent of an angle
__device__ FpDataType htan(const FpDataType angle)
{
  return std::tan(angle);
}

// Give the data type a way to calculate the square root of a value (v)
__device__ inline FpDataType d_sqrt(const FpDataType v) {
    return std::sqrt(v);
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

// A different way to call calculate the tangent of an angle
__device__ inline FpDataType d_tan(const FpDataType a) {
    return htan(a);
}

// A different way to call calculate the base*exponent
__device__ inline FpDataType d_pow(const FpDataType a, const FpDataType b) {
    return hpow(a, b);
}

// The conversion of a float to the data type
__device__ inline FpDataType floatToDataType(const float v) {
    return FpDataType(v);
}

#endif // FP_DATA_TYPE_H