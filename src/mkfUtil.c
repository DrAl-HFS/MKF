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


/***/

static void measurePatternTest (void)
{
   uint8_t patBuf[CELL8_PATTERNS];
   GroupInf inf[CELL8_SYMM_GROUPS];

   LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));
   int nG= c8sGetPattern(patBuf, inf);
   int j= 0, k= 0;
   for (int iG=0; iG<nG; iG++)
   {
      const int n= inf[iG].count;
      LOG("\nG[%d] n=%d: ", iG, n);
      for (int i=0; i<n; i++) { k= patBuf[j+i]; LOG("%d ", gWEP3[k]); }
      j+= n;
   }
   LOG("%s", "\n***\n");
} // measurePatternTest

void mkfuTest (void)
{
   LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));
   measurePatternTest();
} // mkfuTest
