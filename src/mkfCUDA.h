// mkf.cu - Minkowski Functional pattern processing using CUDA
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#ifndef MKF_CUDA_H
#define MKF_CUDA_H

#include "ctUtil.h"

#ifdef __cplusplus
extern "C" {
#endif


#define MC_GT (1<<2)
#define MC_EQ (1<<1)
#define MC_LT (1<<0)

typedef struct
{
   float t[1];
   int   m;
} MKBMapF32;

//typedef size_t MKCount;

#ifndef MKF_CUDA_CU // nvcc doesn't like this prototype...
extern int mkfProcess (const Context *pC, const int def[3], const MKBMapF32 *pMC);
#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_CUDA_H
