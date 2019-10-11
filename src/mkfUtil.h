// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#ifndef MKF_UTIL_H
#define MKF_UTIL_H

#include "util.h"
#include "cell8Sym.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MKF_BINS (1<<8)

//typedef double MKMeasureVal;

// typedef unsigned char MTBits; // SSMMEEAA
// S,M,E bits refer to dimensionality
// S: space 0..3
// M: measure 0..3
// E: estimator 0..3 (punctual, lineal, areal, volumetric)
// A: algorithm used

//typedef struct { double m[4]; MTBits t[4]; } MKMeasureTuple;



/***/
// Compute measures on previously obtained Binary Pattern Distributions.

extern int mkfMeasureBPFD (float m[4], const size_t aBPFD[MKF_BINS], const float s, const int profID);

// Volume fraction
extern float volFrac (const size_t hBPD[MKF_BINS]);

#ifdef MKF_TEST

extern float volFrac8 (const size_t hBPD[MKF_BINS]);

extern void mkfuTest (int m);

#endif

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_UTIL_H
