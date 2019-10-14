// mkfCUDA.h - Minkowski Functional pattern processing using CUDA
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Oct 2019

#ifndef MKF_CUDA_H
#define MKF_CUDA_H

#include "binMapCUDA.h"
#include "mkfUtil.h"
#include "encoding.h"


#ifdef __cplusplus
extern "C" {
#endif

// Just hacky placeholders for now...
#define MKFCU_PROFILE_FAST (3)
#define MKFCU_PROFILE_FLEX (2)

// Device pointers: pBPFD, pW
size_t * mkfCUDAGetBPFD (size_t * pBPFD, const BMOrg *pO, const BMPackWord * pW, const int profile);

// Host result address
size_t * mkfCUDAGetBPFDH (size_t * pBPFD, const BMOrg *pO, const BMPackWord * pW, const int profile);

//extern "C"
int mkfCUDAGetBPFDautoCtx (Context *pC, const FieldDef def[3], const BinMapF64 *pMC, const int profile);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_CUDA_H
