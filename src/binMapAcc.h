// binMapAcc.h - packed binary map generation from scalar field, with (optional) OpenACC support.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef BIN_MAP_ACC_H
#define BIN_MAP_ACC_H

#include "binMapUtil.h"
#include "util.h"
#include "mkfTest.h"    // for compiler pragma warning settings

#ifdef __PGI   // HACKY
#define ACC_INLINE inline
#endif

#ifdef __cplusplus
extern "C" {
#endif


/***/

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

#endif // BIN_MAP_ACC_H

