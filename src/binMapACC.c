// binMapAcc.c - packed binary map generation from scalar field.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "binMapACC.h"

#ifndef ACC_INLINE
#define ACC_INLINE
#endif


/***/

#if 0 // DEPRECATED
// BM_NUMT > 1
// interval threshold (hysteresis)
ACC_INLINE int sgnIdxDiff2F32 (const F32 f, const F32 t[2])
{
   return( sgnIdxDiff1F32(f,t[0]) + 3 * sgnIdxDiff1F32(f,t[1]) );
} // sgnIdxDiff2F32

ACC_INLINE int bm2F32 (const F32 f, const BinMapCtxF32 *pC)
{  // pragma acc data present( pC[:1] )
   return( ( pC->m >> (sgnIdxDiff2F32(f, pC->t[0]) + 3 * sgnIdxDiffF32(f, pC->t[1])) ) & 0x1 );
} // bm2F32
#endif

// '<' '=' '>' -> -1 0 +1
//int sgnF32 (const F32 f) { return((f > 0) - (f < 0)); }

// '<' '=' '>' -> 0, 1, 2
ACC_INLINE int sgnIdxDiff (const MKFAccScalar f, const MKFAccScalar t[BM_NUMT])
{
   return( 1 - (f < t[BM_TLB])) + (f > t[BM_TUB]); // 1+sgnF32(f-t)
} // sgnIdxDiff

ACC_INLINE int binMap (const MKFAccScalar f, const MKFAccBinMap * const pM)
{  // pragma acc data present( pC[:1] )
   return( (pM->m >> sgnIdxDiff(f, pM->t) ) & 0x1 );
} // binMap

#pragma acc routine vector
ACC_INLINE BMPackWord binMapW (const MKFAccScalar * restrict pF, const MKFAccBinMap *pM)
{
   BMPackWord w= 0;
#if 0
   BMPackWord v[32]; // for vectorisation
   #pragma acc loop vector
   for (int j= 0; j < 32; j++) { v[j]= binMap(pF[j], pM) << j; }

   #pragma acc loop reduction(+: w )
   for (int j= 0; j < 32; j++) { w += v[j]; }
#else
   #pragma acc loop reduction(+: w )
   for (int j= 0; j < 32; j++) { w += binMap(pF[j], pM) << j; }
#endif
   return(w);
} // binMapW


/***/

#pragma acc routine vector
void binMapAcc (BMPackWord * restrict pW, const MKFAccScalar * restrict pF, const size_t nF, const MKFAccBinMap *pM)
{  // collapsable fields (rowLen%32 == 0) only!
   const size_t nW= nF>>5; // NB: truncates non multiple of 32!
   #pragma acc data present( pW[:nW], pF[:nF], pM[:1] )
   {
      #pragma acc loop vector
      for (size_t i= 0; i < nW; i++)
      {
         pW[i]= binMapW(pF+i*32, pM);
      }
   }
} // binMapAcc

void binMapRowsAcc
(
   BMPackWord * restrict pW, const MKFAccScalar * restrict pF,
   const int rowLenF, const int rowStrideBM,
   const int nRows, const MKFAccBinMap *pM
)
{
   #pragma acc data present( pW[:rowStrideBM*nRows], pF[:rowLenF*nRows], pM[:1] )
   {
      const int rowLenBits= (rowLenF + 0x1F) & ~0x1F;
      #pragma acc parallel loop
      for (int i= 0; i < nRows; i++)
      {
         binMapAcc(pW + i * rowStrideBM, pF + i * rowLenF, rowLenBits, pM);
      }
   }
} // binMapRowsAcc

// TODO: non planar scalar fields (3D stride)  - may get around to this if need arises...

