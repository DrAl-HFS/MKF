// ctUtil.h - cuda test utils
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#ifndef CT_UTIL_H
#define CT_UTIL_H

#include "util.h"
#include "binMapUtil.h"

#ifdef __cplusplus
extern "C" {
#endif


int ctuInfo (void);

int cuBuffAlloc (Context *pC, uint n);
int cuBuffRelease (Context *pC);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __NVCC__ // MKF_CUDA
extern cudaError_t ctuErr (cudaError_t *pE, const char *s);
#endif

#endif // CT_UTIL_H
