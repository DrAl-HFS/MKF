// mkfCUDA.h - Minkowski Functional pattern processing using CUDA
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Sept 2019

#ifndef MKF_CUDA_H
#define MKF_CUDA_H

#include "binMapCUDA.h"
#include "mkfUtil.h"


#ifdef __cplusplus
extern "C" {
#endif

int mkfCUDAGetBPFD (size_t * pBPFD, const int def[3], const BMStrideDesc *pSD, const U32 * pBM);

//extern "C"
int mkfCUDAGetBPFDautoCtx (Context *pC, const int def[3], const BinMapF32 *pMC);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_CUDA_H
