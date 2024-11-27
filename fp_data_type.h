#ifndef FP_DATA_TYPE_H
#define FP_DATA_TYPE_H

#include <cuda_fp16.h>

typedef __half FpDataType;

#define FP_DATA_TYPE_ZERO FpDataType(0.0f)
#define FP_DATA_TYPE_ONE FpDataType(1.0f)
#define FP_DATA_TYPE_TWO FpDataType(2.0f)
#define FP_DATA_TYPE_THREE FpDataType(3.0f)
#define FP_DATA_TYPE_FOUR FpDataType(4.0f)
#define FP_DATA_TYPE_FIVE FpDataType(5.0f)

__device__ FpDataType hpow(const FpDataType base, const FpDataType power)
{
    return hexp(power * hlog(base));
}

__device__ FpDataType htan(const FpDataType angle)
{
    const FpDataType cos = hcos(angle);
    const FpDataType sin = hsin(angle);
    const FpDataType tolerance = FpDataType(1e-3);
    if (__habs(cos) < tolerance)
    {
        return FP_DATA_TYPE_ZERO;
    }
    return sin / cos;
}

#endif // FP_DATA_TYPE_H