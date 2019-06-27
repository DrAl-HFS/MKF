// mkfTools.c - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "mkfTools.h"

#ifndef ACC_INLINE
#define ACC_INLINE
#endif

/***/

#include "weighting.inc"

/***/

ACC_INLINE void ldbs (U16 bs[4], const U8 * restrict pR0, const U8 * restrict pR1, const U32 rowStride, const U32 lsh)
{  // vect
   bs[0] |= (pR0[0] << lsh);
   bs[1] |= (pR0[rowStride] << lsh);
   bs[2] |= (pR1[0] << lsh);
   bs[3] |= (pR1[rowStride] << lsh);
} // ldbs

ACC_INLINE int mkbp (U8 bp[], U16 bs[4], const int n)
{
   for (int i= 0; i < n; i++) // seq
   {
      U8 r[4];

      for (int k= 0; k < 4; k++) // vect
      {
         r[k]= (bs[k] & 0x3) << (2*k);
         bs[k] >>= 1;
      }
      bp[i]= r[0] | r[1] | r[2] | r[3];
   }
   return(n);
} // mkbp

ACC_INLINE void addRowBPD (U32 hBPD[256], const U8 * restrict pRow[2], const int rowStride, const int n)
{  // seq
   int m, k, j, i;
   U16 bs[4]= { 0,0,0,0 };
   U8 bp[8];

   ldbs(bs, pRow[0]+0, pRow[1]+0, rowStride, 0);
   k= MIN(7, n-1);
   k= mkbp(bp, bs, k);
   for (j=0; j<k; j++) { hBPD[ bp[j] ]++; }
   i= 0;
   m= n>>3;
   while (++i < m)
   {
      ldbs(bs, pRow[0]+i, pRow[1]+i, rowStride, 1);
      k= mkbp(bp, bs, 8);
      for (int j=0; j<k; j++) { hBPD[ bp[j] ]++; }
   }
   k= n & 0x7;
   if (k > 0)
   {
      ldbs(bs, pRow[0]+i, pRow[1]+i, rowStride, 1);
      k= mkbp(bp, bs, k);
      for (int j=0; j<k; j++) { hBPD[ bp[j] ]++; }
   }
} // addRowBPD

void procSimple (U32 hBPD[256], U8 *pBM, const F32 *pF, const int def[3], const BinMapCtxF32 *pC)
{
   const int rowStride= BITS_TO_BYTES(def[0]);
   const int planeStride= rowStride * def[1];
   const int nF= def[0]*def[1]*def[2];

   #pragma acc data present_or_create( pBM[:planeStride*def[2]] ) present_or_copyin( pF[:nF], pC[:1] )
   {  // #pragma acc parallel vector ???
      if ((rowStride<<3) == def[0])
      {  // Multiple of 8
         binMapNF32(pBM, pF, nF, pC);
      }
      else
      {
         binMapRowsF32(pBM, pF, def[0], rowStride, def[1]*def[2], pC);
      }

      #pragma acc data copy( hBPD[:256] )
      {
         for (int j= 0; j < (def[2]-1); j++)
         {
            //#pragma acc parallel vector
            for (int i= 0; i < (def[1]-1); i++)
            {
               const U8 * pRow[2];
               pRow[0]= pBM + i * rowStride + j * planeStride;
               pRow[1]= pRow[0] + planeStride;
               addRowBPD(hBPD, pRow, rowStride, def[0]);
            }
         }
      }
   }
   { // debug...
      size_t t= 0;
      for (int i= 0; i < 256; i++)
      {
         //LOG("%d: %u\n", i, hBPD[i]);
         t+= hBPD[i] * bitCountZ(i);
      }
      LOG("bitcount=%zu /8= %zu\n", t, t>>3);
   }
} // procSimple


/***/

MKMeasureVal volFrac (const U32 h[256])
{
   size_t s[2]={0,0};
   for (int i= 0; i<256; i+= 2)
   {
      s[0]+= h[i];
      s[1]+= h[i+1];
   }
   LOG_CALL("() - s[]={%zu, %zu} (%zu)\n", s[0], s[1], s[0]+s[1]);
   return( s[1] / (MKMeasureVal)(s[0] + s[1]) );
} // volFrac

MKMeasureVal chiEP3 (const U32 h[256])
{
   I32 k=0;
   for (int i= 0; i<256; i++) { k+= (I32)gWEP3[i] * (I32)h[i]; }
   //LOG_CALL("() - k[]={%i, %i}\n", k[0], k[1]);
   return( (MKMeasureVal) k * M_PI / 6 );
} // chiEP3


