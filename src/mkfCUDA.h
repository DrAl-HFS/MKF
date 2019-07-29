// mkfCUDA.h - Minkowski Functional pattern processing using CUDA
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#ifndef MKF_CUDA_H
#define MKF_CUDA_H

#include "binMapCUDA.h"
#include "mkfUtil.h"


#ifndef MKF_CUDA_CU // nvcc doesn't like this prototype...
extern "C" int mkfCUDAGetBPFDSimple (const Context *pC, const int def[3], const BinMapF32 *pMC);
#endif

#endif // MKF_CUDA_H
