// binMapAcc.c - packed binary map generation from scalar field.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "binMapAcc.h"

#ifndef ACC_INLINE
#define ACC_INLINE
#endif


/***/

// '<' '=' '>' -> -1 0 +1
//int sgnF32 (const F32 f) { return((f > 0) - (f < 0)); }

// '<' '=' '>' -> 0, 1, 2
ACC_INLINE U32 sgnIdxDiff1F32 (const F32 f, const F32 t) { return(1 + (f > t) - (f < t)); } // 1+sgnF32(f-t)

ACC_INLINE U32 bm1F32 (const F32 f, const BinMapF32 * const pC)
{  // pragma acc data present( pC[:1] )
   return( (pC->m >> sgnIdxDiff1F32(f, pC->t[0]) ) & 0x1 );
} // bm1F32

#if BM_NUMT > 1
// interval threshold (hysteresis)
ACC_INLINE U32 sgnIdxDiff2F32 (const F32 f, const F32 t[2])
{
   return( sgnIdxDiff1F32(f,t[0]) + 3 * sgnIdxDiff1F32(f,t[1]) );
} // sgnIdxDiff2F32

ACC_INLINE U8 bm2F32 (const F32 f, const BinMapCtxF32 *pC)
{  // pragma acc data present( pC[:1] )
   return( ( pC->m >> (sgnIdxDiff2F32(f, pC->t[0]) + 3 * sgnIdxDiffF32(f, pC->t[1])) ) & 0x1 );
} // bm2F32
#endif


/***/

#pragma acc routine vector
void binMapNF32G (U32 * restrict pBM, const F32 * restrict pF, const size_t n, const BinMapF32 * const pC)
{
   #pragma acc data present( pBM[:n], pF[:n], pC[:1] )
   {
      #pragma acc loop vector
      for (size_t i= 0; i < n; i++)
      { //
         pBM[i]= bm1F32(pF[i], pC); // << (i & 0x7);
      }
   }
} // binMapNF32G

#pragma acc routine vector
void binMapNF32 (U32 * restrict pBM, const F32 * restrict pF, const size_t nF, const BinMapF32 *pC)
{
   const size_t nB= nF>>5; // NB: truncates non multiple of 8!
   #pragma acc data present( pBM[:nB], pF[:nF], pC[:1] )
   {
      #pragma acc loop vector
      for (size_t i= 0; i < nB; i++)
      { //
         U32 b= 0;
         #pragma acc loop seq
         for (int j= 0; j < 32; j++) { b|= bm1F32(pF[(i*32)+j], pC) << j; }
         pBM[i]= b;
      }
   }
} // binMapNF32

void binMapRowsF32
(
   U32 * restrict pBM, const F32 * restrict pF,
   const int rowLenF, const int rowStrideBM,
   const int nRows, const BinMapF32 *pC
)
{
   #pragma acc data present( pBM[:rowStrideBM*nRows], pF[:rowLenF*nRows], pC[:1] )
   {
      const int rowLenBits= (rowLenF + 0x1F) & ~0x1F;
      #pragma acc loop vector
      for (int i= 0; i < nRows; i++)
      {
         binMapNF32(pBM + i * rowStrideBM, pF + i * rowLenF, rowLenBits, pC);
      }
   }
} // binMapRowsF32

// TODO: non planar scalar fields (3D stride) ?

#if 0

#include "openacc.h"

void binMapInit (void)
{  // Only multicore acceleration works presently: GPU produces garbage...
   acc_set_device_num( 0, acc_device_host );
}

#endif

