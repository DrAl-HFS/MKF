// mkfTools.h - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef MKF_TOOLS_H
#define MKF_TOOLS_H

#include "binMap.h"

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
extern void procSimple (U32 rBPD[256], U32 *pBM, const F32 *pF, const int def[3], const BinMapF32 *pC);


/***/
// Compute measures on previously obtained Binary Pattern Distributions.

// Volume fraction
extern MKMeasureVal volFrac (const U32 hBPD[256]);

// Euler-Poincare Chi for 3D
extern MKMeasureVal chiEP3 (const U32 hBPD[256]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_TOOLS_H
