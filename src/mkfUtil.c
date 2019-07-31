// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "mkfUtil.h"


/***/

#include "weighting.inc"


/***/

MKMeasureVal volFrac (const size_t aBPFD[MKF_BINS])
{
   size_t s[2]={0,0};
   for (int i= 0; i<MKF_BINS; i+= 2)
   {
      s[0]+= aBPFD[i];
      s[1]+= aBPFD[i+1];
   }
   LOG_CALL("() - s[]={%zu, %zu} (%zu)\n", s[0], s[1], s[0]+s[1]);
   return( s[1] / (MKMeasureVal)(s[0] + s[1]) );
} // volFrac

MKMeasureVal chiEP3 (const size_t aBPFD[MKF_BINS])
{
   I64 k=0;
   for (int i= 0; i<MKF_BINS; i++) { k+= (I64)gWEP3[i] * (signed)aBPFD[i]; }
   //LOG_CALL("() - k[]={%i, %i}\n", k[0], k[1]);
   return( (MKMeasureVal) k * M_PI / 6 );
} // chiEP3


void symTst (void)
{
   U16 s;
   U8 m[1<<8];
   U8 s0, t0, n=0;

   for (int i=0; i<256; i++) { m[i]= i; }
   s= t0= 0x01;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 1;
   } while (s < 256);
   s= t0= 0x03;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 2;
   } while (s < 256);
   s= t0= 0x05;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 2;
   } while (s < 256);
   s= t0= 0x0A;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 2;
   } while (s < 256);
   s= t0= 0x0F;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 4;
   } while (s < 256);
   for (int i=0; i<256; i++)
   {
      if (i != m[i]) { LOG("[%02X]=%02X\n", i, m[i]); n++; }
   }
   LOG("n=%u\n", n);
} // symTst


void mkfuTest (void)
{
   U8 d[33]={0};
   //rangeNI(mm,
   for (int i= 0; i<256; i++)
   {
      U8 b= 16+gWEP3[i];
      d[b]++;
   }
   for (int i= -16; i<=16; i++)
   {
      U8 n= d[16+i];
      if (0 != n) { LOG("[%d] %u\n", i, n); }
   }
} // mkfuTest
