// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "mkfUtil.h"
#include "cell8Sym.h"


/***/

#include "weighting.inc"
#ifndef NO_REF_MEASURES
#include "refMeasures.inc"
#endif

static const int8_t gWTC[CELL8_SYMM_GROUPS]={0, 3, 0, 6, 6, -3, 3, 9, 0, -6, -6, 0, 12, 0, -3, -8, -3, -12, -6, 0, 3, 0};

static uint8_t c8sMap[CELL8_PATTERNS];

//int refMeasures (float m[4], const size_t aBPFD[MKF_BINS], const float s) { ... }

/***/

size_t sumNZ (const size_t a[], const int n)
{
   size_t s= a[0];   // assumes n>0 !
   for (int i=1; i<n; i++) { s+= a[i]; }
   return(s);
} // sumNZ

static void addZMapU8 (size_t r[], const size_t vA[], const int nA, const uint8_t map[])
{
// size_t s=0;
   for (int i= 0; i<nA; i++)
   {
      r[ map[i] ]+= vA[i]; // s+= vA[i];
   }
} // addZMapU8

float tc (const size_t rBPFD[CELL8_SYMM_GROUPS])
{
   I64 k=0;
   for (int i= 0; i<CELL8_SYMM_GROUPS; i++) { k+= gWTC[i] * (signed)rBPFD[i]; }
   return( (float) k * M_PI / 6 );
} // tc

/***/

int mkfMeasureBPD (float m[4], const size_t aBPFD[MKF_BINS], const float s, const int profID)
{
   switch (profID)
   {
#ifndef NO_REF_MEASURES
      case 0 : return refMeasures(m, aBPFD, s);
#endif
      default :
      {
         size_t t, rBPFD[CELL8_PATTERNS]={0,};
         t= sumNZ(aBPFD, MKF_BINS);
         if (0x01 != c8sMap[1]) { c8sGetMap(c8sMap); } // Lazy init
         addZMapU8(rBPFD, aBPFD, MKF_BINS, c8sMap);
         m[0]= tc(rBPFD);
         if ((t == sumNZ(rBPFD, CELL8_PATTERNS)) && (s > 0))
         {
            float rkV= 1.0 / (s * s * s * t); //sumNZ(rBPFD, CELL8_PATTERNS);
            m[0]*= rkV;
         }
         m[1]= m[2]= m[3]= -1;
         return(1);
      }
   }
   return(0);
} // mkfMeasureBPD

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
   int8_t wK[CELL8_SYMM_GROUPS];

   LOG_CALL("() [FP=%p] GroupInf=%d\n",__builtin_frame_address(0), sizeof(GroupInf));
   int nG= c8sGetPattern(patBuf, inf);
   int j= 0, k= 0;
   for (int iG=0; iG<nG; iG++)
   {
      const int n= inf[iG].count;
      LOG("\nG[%d] n=%d: ", iG, n);
      wK[iG]= gWEP3[ patBuf[j] ];
      for (int i=0; i<n; i++) { k= patBuf[j+i]; LOG("%d ", gWEP3[k]); }
      j+= n;
   }
   LOG("%s", "\n\n");
   for (int iG=0; iG<nG; iG++) { LOG("%d, ", wK[iG]); }

   LOG("%s", "\n***\n");
} // measurePatternTest

void mkfuTest (void)
{
   LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));
   //c8sTest();
   measurePatternTest();
} // mkfuTest
