#ifndef FP_DATA_TYPE_H
#define FP_DATA_TYPE_H

// #include <cuda_fp16.h>

typedef double FpDataType;

#define FP_DATA_TYPE_ZERO FpDataType(0.0f)
#define FP_DATA_TYPE_ONE FpDataType(1.0f)
#define FP_DATA_TYPE_TWO FpDataType(2.0f)
#define FP_DATA_TYPE_THREE FpDataType(3.0f)
#define FP_DATA_TYPE_FOUR FpDataType(4.0f)
#define FP_DATA_TYPE_FIVE FpDataType(5.0f)

__device__ FpDataType hpow(const FpDataType base, const FpDataType power)
{
    return exp(power * log(base));
}

__device__ FpDataType htan(const FpDataType angle)
{
    const FpDataType cos = std::cos(angle);
    const FpDataType sin = std::sin(angle);
    const FpDataType tolerance = FpDataType(1e-3);
    if (abs(cos) < tolerance)
    {
        return FP_DATA_TYPE_ZERO;
    }
    return sin / cos;
}

__device__ inline FpDataType d_sqrt(const FpDataType v) {
    return std::sqrt(v);
}

__device__ inline FpDataType d_mul(const FpDataType a, const FpDataType b) {
    return a * b;
}

__device__ inline FpDataType d_div(const FpDataType a, const FpDataType b) {
    return a / b;
}

__device__ inline FpDataType d_add(const FpDataType a, const FpDataType b) {
    return a + b;
}

__device__ inline FpDataType d_sub(const FpDataType a, const FpDataType b) {
    return a - b;
}

__device__ inline FpDataType d_neg(const FpDataType a) {
    return a * -1;
}

__device__ inline FpDataType d_tan(const FpDataType a) {
    return std::tan(a);
}

__device__ inline FpDataType floatToDataType(const float v) {
    return FpDataType(v);
}

#endif // FP_DATA_TYPE_H