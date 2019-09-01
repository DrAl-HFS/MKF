// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "mkfUtil.h"
#include "cell8Sym.h"


/***/

#include "weighting.inc"
#include "refMeasures.inc"

//int refMeasures (float m[4], const size_t aBPFD[MKF_BINS], const float s) { ... }

/***/

float volFrac (const size_t aBPFD[MKF_BINS])
{
   size_t s[2]={0,0};
   for (int i= 0; i<MKF_BINS; i+= 2)
   {
      s[0]+= aBPFD[i];
      s[1]+= aBPFD[i+1];
   }
   LOG_CALL("() - s[]={%zu, %zu} (%zu)\n", s[0], s[1], s[0]+s[1]);
   return( s[1] / (float)(s[0] + s[1]) );
} // volFrac

float volFrac8 (const size_t aBPFD[MKF_BINS])
{
   size_t s[2]={0,0};
   for (int i= 0; i<MKF_BINS; i++)
   {
      s[0]+= aBPFD[i] * bitCountZ(i);
      s[1]+= aBPFD[i];
   }
   s[1]*= 8;
   LOG_CALL("() - s[]={%zu, %zu}\n", s[0], s[1]);
   return( s[0] / (float)s[1] );
} // volFrac

float chiEP3 (const size_t aBPFD[MKF_BINS])
{
   I64 k=0;
   for (int i= 0; i<MKF_BINS; i++) { k+= (I64)gWEP3[i] * (signed)aBPFD[i]; }
   //LOG_CALL("() - k[]={%i, %i}\n", k[0], k[1]);
   return( (float) k * M_PI / 6 );
} // chiEP3

void measurePatternTest (void)
{
   uint8_t patBuf[256];
   GroupInf inf[32];

   int nG= c8sGetPattern(patBuf, inf);
   int j= 0, k= 0;
   for (int iG=0; iG<nG; iG++)
   {
      const int n= inf[iG].count;
      LOG("\n%d: ", n);
      for (int i=0; i<n; i++) { k= patBuf[j+i]; LOG("%d ", gWEP3[k]); }
      j+= n;
   }
   LOG("%s", "\n***\n");
} // measurePatternTest

void mkfuTest (void)
{
   return measurePatternTest();
   U8 d[33]={0};
   //I8 v[16], nV=0;
   int s=0;
   //rangeNI(mm,
   for (int i= 0; i<256; i++)
   {
      U8 b= 16+gWEP3[i];
      d[b]++;
   }
   for (int i= -16; i<=16; i++)
   {
      U8 z, lz=-1;
      U8 n= d[16+i];
      if (0 != n)
      {
         //v[nV++]= i;
         LOG("\n%+d : [%u]= ", i, n);
         s+= n;
         for (int j= 0; j<256; j++)
         {
            if (gWEP3[j] == i)
            {
               LOG("%02X ", j);
               z= bitCountZ(j);
               if (z != lz)
               {
                  lz= z;
                  LOG("(z=%u) ", z);
               }
            }
         }
      }
   }
   LOG("\ns=%d\n***\n", s);
} // mkfuTest
