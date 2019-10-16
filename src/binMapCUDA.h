// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Oct 2019

#ifndef BIN_MAP_CUDA_H
#define BIN_MAP_CUDA_H

#include "ctUtil.h"
#include "binMapUtil.h"

#ifdef __cplusplus
extern "C" {
#endif

BMPackWord *binMapCUDA
(
   KernInfo * pK, // optional
   BMPackWord        * pW,
   BMOrg             * pO,
   const BMFieldInfo * pF,
   const BinMapF64   * pM
);

void binMapCUDACleanup (void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BIN_MAP_CUDA_H
