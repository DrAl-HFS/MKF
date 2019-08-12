// binMapUtil.c - packed binary map general utility code.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "binMapUtil.h"


/***/

#if 0
// '<' '=' '>' -> -1 0 +1
//int sgnF32 (const F32 f) { return((f > 0) - (f < 0)); }

// '<' '=' '>' -> 0, 1, 2
U32 sgnIdxDiff1F32 (const F32 f, const F32 t) { return(1 + (f > t) - (f < t)); } // 1+sgnF32(f-t)

U32 bm1F32 (const F32 f, const BinMapF32 * const pC)
{  // pragma acc data present( pC[:1] )
   return( (pC->m >> sgnIdxDiff1F32(f, pC->t[0]) ) & 0x1 );
} // bm1F32

// interval threshold (hysteresis)
U32 sgnIdxDiff2F32 (const F32 f, const F32 t[2])
{
   return( sgnIdxDiff1F32(f,t[0]) + 3 * sgnIdxDiff1F32(f,t[1]) );
} // sgnIdxDiff2F32

U8 bm2F32 (const F32 f, const BinMapCtxF32 *pC)
{  // pragma acc data present( pC[:1] )
   return( ( pC->m >> (sgnIdxDiff2F32(f, pC->t[0]) + 3 * sgnIdxDiffF32(f, pC->t[1])) ) & 0x1 );
} // bm2F32

#endif

/***/

void setBinMapF32 (BinMapF32 *pC, const char relopChars[], const float t)
{
   int i= 0, inv= 0, valid= 1;
   pC->t[0]= t;
   pC->m= BMC_NV;

   do
   {
      switch(relopChars[i])
      {
         case '<' : pC->m|= BMC_LT; break;
         case '=' : pC->m|= BMC_EQ; break;
         case '>' : pC->m|= BMC_GT; break;
         case '!' : if (0 == i) { inv= 1; break; } // else...
         default : valid= 0;
      }
      ++i;
   } while (valid);
   if (inv) { pC->m^= BMC_AL; }
   //LOG_CALL("(%p, %s, %G) - m=0x%X\n", pC, relopChars, t, pC->m);
} // setBinMapF32

size_t setBMSD (BMStrideDesc *pSD, const int def[3], const char profID)
{
   BMStrideDesc sd;
   uint maxPlane= MIN(2, def[2]);

   if (NULL == pSD) { pSD= &sd; }
   sd.row= BITS_TO_WRDSH(def[0],5);  // Packed 32bit words
   switch (profID)
   {
      case 0 :
         maxPlane= def[2];
         break;
   }
   sd.plane= sd.row * def[1];
   if (pSD) { *pSD= sd; }
   return((size_t)sd.plane * maxPlane);
} // setBMSD

#if 0
testBMC (const float f0, const float fs, const int n, const BinMapF32 *pC)
{
   for (int i= 0; i<n; i++)
   {
      F32 f= f0 + i * fs;
      U8 r= bm1F32(f, &ctx);
      LOG("\t%G -> %u\n", f, r);
   }
}
#endif

