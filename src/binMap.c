// binMap.c - packed binary map generation from scalar field.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "binMap.h"

#ifndef ACC_INLINE
#define ACC_INLINE
#endif

/***/

// '<' '=' '>' -> -1 0 +1
//int sgnF32 (const F32 f) { return((f > 0) - (f < 0)); }

// '<' '=' '>' -> 0, 1, 2
ACC_INLINE U8 sgnIdxDiffF32 (const F32 f, const F32 t) { return(1 + (f > t) - (f < t)); } // 1+sgnF32(f-t)

ACC_INLINE U8 bm1F32 (const F32 f, const BinMapCtxF32 *pC)
{  // pragma acc data present( pC[:1] )
   return( pC->m[ sgnIdxDiffF32(f, pC->t[0]) ] );
} // bm1F32

/* interval threshold (hysteresis)
ACC_INLINE U8 bm2F32 (const F32 f, const BinMapCtxF32 *pC)
{  // pragma acc data present( pC[:1] )
   return( pC->m[ sgnIdxDiffF32(f, pC->t[0]) + 3 * sgnIdxDiffF32(f, pC->t[1]) ] );
} // bm2F32
*/

/***/

void setBMCF32 (BinMapCtxF32 *pC, const char relopChars[], const F32 t)
{
   int i= 0, valid= 1;
   pC->t[0]= t;
   pC->w= 0;

   while (valid)
   {
      switch(relopChars[i])
      {
         case '<' : pC->m[0]|= 0x01; break;
         case '=' : pC->m[1]|= 0x01; break;
         case '>' : pC->m[2]|= 0x01; break;
         case '!' : if (0 ==i) { pC->m[3]|= 0x80; break; } // else...
         default : valid= 0;
      }
      ++i;
   }
   if (pC->m[3] & 0x80) { for (i= 0; i < 3; i++) { pC->m[i]^= 0x01; } }
   LOG_CALL("(%p, %s, %G) - m={0x%X:0x%X:0x%X:0x%X}\n", pC, relopChars, t, pC->m[0], pC->m[1], pC->m[2], pC->m[3]);
} // setBMCF32

#pragma acc routine
void binMapNF32 (U8 * restrict pBM, const F32 * restrict pF, const size_t nF, const BinMapCtxF32 *pC)
{
   #pragma acc data present( pBM[:BITS_TO_BYTES(nF)], pF[:nF], pC[:1] )
   {  // #pragma acc parallel vector ???
      for (size_t i= 0; i < nF; i+= 8)
      { //#pragma acc seq ???
         U8 b= 0;
         for (int j= 0; j < 8; j++) { b|= bm1F32(pF[i+j], pC) << j; }
         pBM[i >> 3]= b;
      }
   }
} // binMapNF32

void binMapRowsF32
(
   U8 * restrict pBM, const F32 * restrict pF,
   const int rowLenF, const int rowStrideBM,
   const int nRows, const BinMapCtxF32 *pC
)
{
   #pragma acc data present( pBM[:rowStrideBM*nRows], pF[:rowLenF*nRows], pC[:1] )
   {
      const int rowLenBits= (rowLenF + 0x7) & ~0x7;
      #pragma acc parallel
      for (int i= 0; i < nRows; i++)
      {
         binMapNF32(pBM + i * rowStrideBM, pF + i * rowLenF, rowLenBits, pC);
      }
   }
} // binMapRowsF32

// TODO: non planar scalar fields (3D stride) ?


#if 0
testBMC (const F32 f0, const F32 fs, const int n, const BinMapCtxF32 *pC)
{
   for (int i= 0; i<n; i++)
   {
      F32 f= f0 + i * fs;
      U8 r= bmF32(f, &ctx);
      LOG("\t%G -> %u\n", f, r);
   }
}
#endif

