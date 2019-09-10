// mkfUtil.c - tools for calculating Minkowski Functionals from pattern distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#include "mkfUtil.h"
#include "cell8Sym.h"
#include "mkfHacks.h"

/***/

#ifndef NO_REF_MEASURES
// Reference measures published in ""
#include "refMeasures.inc"
#endif

static const int8_t gWK[CELL8_SYMM_GROUPS]=
{  0, 3, 0, 6, 6, -3, 3,
   9, 0, -6, -6, 0, 12, 0,
   -3, -8, -3, -12, -6, 0, 3,
   0
};
static uint8_t c8sMap[CELL8_PATTERNS];

//int refMeasures (float m[4], const size_t aBPFD[MKF_BINS], const float s) { ... }

/***/

double symK (const size_t rBPFD[CELL8_SYMM_GROUPS])
{
   I64 k=0;
   for (int i= 0; i<CELL8_SYMM_GROUPS; i++) { k+= gWK[i] * (signed)rBPFD[i]; }
   return( (double) k * M_PI / 6 );
} // symK


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


/***/

int mkfMeasureBPFD (float m[4], const size_t aBPFD[MKF_BINS], const float s, const int profID)
{
   switch (profID)
   {
#ifndef NO_REF_MEASURES
      case 0 : return refMeasures(m, aBPFD, s);
#endif
      default :
      {
         size_t t, rBPFD[CELL8_SYMM_GROUPS]={0,};
         t= sumNZ(aBPFD, MKF_BINS);
         if (0x01 != c8sMap[1]) { c8sGetMap(c8sMap); } // Lazy init
         addZMapU8(rBPFD, aBPFD, MKF_BINS, c8sMap);
         m[0]= symK(rBPFD); // symMS(m+1, rBPFD);
         m[1]= m[2]= m[3]= -1;
         if ((t == sumNZ(rBPFD, CELL8_SYMM_GROUPS)) && (s > 0))
         {
            float rkV= 1.0 / (s * s * s * t); //sumNZ(rBPFD, CELL8_PATTERNS);
            m[0]*= rkV;
            //m[1]*= s;
            //m[2]*= s * s;
         }
         return(1);
      }
   }
   return(0);
} // mkfMeasureBPFD

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



/***/

void mkfuTest (int m)
{
   LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));
   if (m & 0x0F) { c8sTest(); }
   if (m & 0xF0) { measurePatternTest(m >> 4); }
} // mkfuTest
