// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef MKF_UTIL_H
#define MKF_UTIL_H

#include "report.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef double MKMeasureVal;

//typedef struct { MKMeasureVal v, k; } MKVK;


/***/
// Compute measures on previously obtained Binary Pattern Distributions.

// Volume fraction
extern MKMeasureVal volFrac (const U32 hBPD[256]);

// Euler-Poincare Chi for 3D
extern MKMeasureVal chiEP3 (const U32 hBPD[256]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_UTIL_H
