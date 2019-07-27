// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#ifndef BIN_MAP_CUDA_H
#define BIN_MAP_CUDA_H

#include "ctUtil.h"
#include "binMapUtil.h"

__device__ int bm1f32 (const float f, const BinMapF32& bm);

//extern
//__global__ void vThresh32 (uint r[], const float f[], const size_t n, const BinMapF32 mc);

#endif // BIN_MAP_CUDA_H
