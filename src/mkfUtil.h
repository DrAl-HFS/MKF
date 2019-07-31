// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef MKF_UTIL_H
#define MKF_UTIL_H

#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MKF_BINS (1<<8)

typedef double MKMeasureVal;

//typedef struct { MKMeasureVal v, k; } MKVK;


/***/
// Compute measures on previously obtained Binary Pattern Distributions.

// Volume fraction
extern MKMeasureVal volFrac (const size_t hBPD[MKF_BINS]);

// Euler-Poincare Chi for 3D
extern MKMeasureVal chiEP3 (const size_t hBPD[MKF_BINS]);

extern void mkfuTest (void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_UTIL_H
