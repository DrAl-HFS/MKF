
#include "binMap.h"



/***/


#if 0
{
   for (int i= 0; i<5; i++)
   {
      F32 f= i * 0.25;
      U8 r= bmF32(f, &ctx);
      LOG("\t%G -> %u\n", f, r);
   }
}
#endif

// '<' '=' '>' -> -1 0 +1
//int sgnF32 (const F32 f) { return((f > 0) - (f < 0)); }

// '<' '=' '>' -> 0, 1, 2
static U8 sgnIdxDiffF32 (const F32 f, const F32 t) { return(1 + (f > t) - (f < t)); } // 1+sgnF32(f-t)

static U8 bmF32 (const F32 f, const BinMapCtxF32 *pC) { return( pC->m[ sgnIdxDiffF32(f, pC->t[0]) ] ); }


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

void binMapNF32 (U8 * restrict pBM, const F32 * restrict pF, const size_t nF, const BinMapCtxF32 *pC)
{
   #pragma acc parallel vector
   for (size_t i= 0; i < nF; i+= 8)
   { //#pragma acc seq ???
      U8 b= 0;
      for (int j= 0; j < 8; j++) { b|= bmF32(pF[i+j], pC) << j; }
      pBM[i >> 3]= b;
   }
   //return(r);
} // binMapNF32

void binMapRowsF32
(
   U8 * restrict pBM, const F32 * restrict pF,
   const int rowLenF, const int rowStrideBM,
   const int nRows, const BinMapCtxF32 *pC
)
{
   const int rowLenBits= (rowLenF + 0x7) & ~0x7;
   #pragma acc parallel vector
   for (int i= 0; i < nRows; i++)
   {
      binMapNF32(pBM + i * rowStrideBM, pF + i * rowLenF, rowLenBits, pC);
   }
} // binMapRowsF32



