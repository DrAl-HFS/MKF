
#ifndef BINMAP_H
#define BINMAP_H

#include "util.h"
#include "mkfTest.h"


#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   float t[1];
   union { U8 m[4]; U32 w; };
} BinMapCtxF32;

/***/

extern void setBMCF32 (BinMapCtxF32 *pC, const char relopChars[], const F32 t);

extern void binMapNF32 (U8 * restrict pBM, const F32 * restrict pF, const size_t nF, const BinMapCtxF32 *pC);

extern void binMapRowsF32
(
   U8 * restrict pBM, const F32 * restrict pF,
   const int rowLenF, const int rowStrideBM,
   const int nRows, const BinMapCtxF32 *pC
);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BINMAP_H

