// binMap.h - packed binary map generation from scalar field.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef BINMAP_H
#define BINMAP_H

#include "util.h"
#include "mkfTest.h"

#ifdef __PGI   // HACKY
#define ACC_INLINE inline
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Comparison operation descriptor (single threshold).
// The possible 0 or 1 result of a comparison
// is obtained from m[] via a comparison-to-index
// function. Thus any relation < <= = != >= > can
// be described as a combination of three bits.
// The runtime cost is always two comparisons per
// threshold value plus the lookup.
typedef struct
{
   float t[1];
   U32 m;
} BinMapF32;

#define BMC_GT (0x4)
#define BMC_EQ (0x2)
#define BMC_LT (0x1)
#define BMC_NV (0x0)
#define BMC_AL (0x7)


/***/

// Define map (comparison operator) using a string
extern void setBMCF32 (BinMapF32 *pC, const char relopChars[], const F32 t);

// Apply a map without considering alignment - assumes row length is a multiple of 8.
// nF is the product of scalar field (X, Y, Z) "dimensions": this number of packed bits will be generated.
extern void binMapNF32 (U32 * restrict pBM, const F32 * restrict pF, const size_t nF, const BinMapF32 *pC);

// Apply a map to a planar organised 32bit Scalar field.
// Ensures row alignment of the resulting packed binary map.
// Caller to ensure: rowStrideBM >= BITS_TO_BYTES(rowLenF)
extern void binMapRowsF32
(
   U32 * restrict pBM,      // destination for packed binary map
   const F32 * restrict pF,// source scalar field
   const int rowLenF,      // row length ie. "X dimension"
   const int rowStrideBM,  // Byte stride of rows of packed binary map, should be >=
   const int nRows,        // product of other "dimensions" (Y * Z)
   const BinMapF32 *pC
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BINMAP_H

