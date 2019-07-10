// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "mkfUtil.h"


/***/

#include "weighting.inc"

/***/

MKMeasureVal volFrac (const U32 aBPD[256])
{
   size_t s[2]={0,0};
   for (int i= 0; i<256; i+= 2)
   {
      s[0]+= aBPD[i];
      s[1]+= aBPD[i+1];
   }
   LOG_CALL("() - s[]={%zu, %zu} (%zu)\n", s[0], s[1], s[0]+s[1]);
   return( s[1] / (MKMeasureVal)(s[0] + s[1]) );
} // volFrac

MKMeasureVal chiEP3 (const U32 aBPD[256])
{
   I32 k=0;
   for (int i= 0; i<256; i++) { k+= (I32)gWEP3[i] * (I32)aBPD[i]; }
   //LOG_CALL("() - k[]={%i, %i}\n", k[0], k[1]);
   return( (MKMeasureVal) k * M_PI / 6 );
} // chiEP3


