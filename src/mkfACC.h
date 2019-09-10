// mkfAcc.h - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#ifndef MKF_ACC_H
#define MKF_ACC_H

#include "binMapACC.h"
#include "mkfUtil.h"

#ifdef __cplusplus
extern "C" {
#endif


/***/

// Simple version for basic testing
extern int mkfAccGetBPFDSimple (size_t rBPFD[MKF_BINS], U32 * restrict pBM, const F32 * restrict pF, const int def[3], const BinMapF32 *pC);

#ifdef MKF_ACC_CUDA_INTEROP

extern int mkfAccCUDAGetBPFD (size_t rBPFD[MKF_BINS], U32 * pBM, const F32 * pF, const int def[3], const BinMapF32 * const pC);

#endif // MKF_ACC_CUDA_INTEROP

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_ACC_H
