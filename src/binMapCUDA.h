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

//extern "C"
void binMapCudaRowsF32
(
   U32 * pBM,
   const F32 * pF,
   const int rowLenF,      // row length ie. "X dimension"
   const int rowStrideBM,  // 32bit word stride of rows of packed binary map, should be >=
   const int nRows,        // product of other "dimensions" (Y * Z)
   const BinMapF32 *pC
);

BMStrideDesc *binMapCUDA
(
   uint           * pBM,
   BMStrideDesc  * pBMSD,
   const MultiFieldInfo * pMFI,
   const BinMapF32       * pMC
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BIN_MAP_CUDA_H
