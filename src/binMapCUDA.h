// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Sept 2019

#ifndef BIN_MAP_CUDA_H
#define BIN_MAP_CUDA_H

#include "ctUtil.h"
#include "binMapUtil.h"

#ifdef __cplusplus
extern "C" {
#endif

BMStrideDesc *binMapCUDA
(
   BMPackWord        * pBM,
   BMStrideDesc      * pBMSD,
   const BMFieldInfo * pBMFI,
   const BinMapF32   * pMC
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BIN_MAP_CUDA_H
