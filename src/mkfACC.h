// mkfAcc.h - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef MKF_ACC_H
#define MKF_ACC_H

#include "binMapACC.h"
#include "mkfUtil.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef double MKMeasureVal;

//typedef struct { MKMeasureVal v, k; } MKVK;

/***/
// Generate Binary Pattern Distribution of a 3D scalar field ("volume image") for subsequent measurement.
// An intermediate packed-binary (in reverse network bit order) image is generated. For row
// length not a multiple of 8, each row is padded to a whole byte (i.e. every row always starts
// at a byte address)

//extern int addRowBPD (U32 hBPD[256], const U8 * restrict pRow[2], const int rowStride, const int n);

// Simple version for basic testing
extern int mkfAccGetBPFDSimple (size_t rBPFD[MKF_BINS], U32 * restrict pBM, const F32 * restrict pF, const int def[3], const BinMapF32 *pC);

#ifdef MKF_ACC_CUDA_INTEROP

extern int mkfAccCUDAGetBPFD (size_t rBPFD[MKF_BINS], U32 * pBM, const F32 * pF, const int def[3], const BinMapF32 * const pC);

#endif // MKF_ACC_CUDA_INTEROP

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_ACC_H
