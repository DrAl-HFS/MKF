// mkfUtil.c - tools for calculating Minkowski Functionals from Binary Pattern Frequency Distribution.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#include "mkfUtil.h"
#include "geomSVU.h"
#ifdef MKF_TEST
#include "cell8Sym.h"
#include "mkfHacks.h"
#endif


/***/

#ifndef NO_REF_MEASURES

// Source code for reference measures published in:
// Ohser J, Mucklich F. (2001) "Statistical Analysis of
//    Microstructures in Materials Science" pp.116-123,
//    Wiley, ISBN-0471974862.

#include "refMeasures.c"

// Other references:
// Serra J, (1983) "Image Analysis and Mathematical Morphology, Volume 1"
//    Academic Press (London)
// Matheron G. (1967) "Elements pour une Theorie des Milieux Poreux"
//    Masson et Cie. (Paris)
#endif

// Topological (total curvature) measure weights
// for cubic cell symmetry groups
static const int8_t gWK[CELL8_SYMM_GROUPS]=
{  0, 3, 0, 6, 6, -3, 3,
   9, 0, -6, -6, 0, 12, 0,
   -3, -8, -3, -12, -6, 0, 3,
   0
};
// Mapping from BPFD to symmetry groups i.e. the group index for each possible pattern
static uint8_t c8sMap[CELL8_PATTERNS]=
{
   0x00,0x01,0x01,0x02,0x01,0x02,0x03,0x05,0x01,0x03,0x02,0x05,0x02,0x05,0x05,0x08,
   0x01,0x02,0x03,0x05,0x03,0x05,0x07,0x09,0x04,0x06,0x06,0x0A,0x06,0x0A,0x0D,0x10,
   0x01,0x03,0x02,0x05,0x04,0x06,0x06,0x0A,0x03,0x07,0x05,0x09,0x06,0x0D,0x0A,0x10,
   0x02,0x05,0x05,0x08,0x06,0x0A,0x0D,0x10,0x06,0x0D,0x0A,0x10,0x0B,0x0F,0x0F,0x13,
   0x01,0x03,0x04,0x06,0x02,0x05,0x06,0x0A,0x03,0x07,0x06,0x0D,0x05,0x09,0x0A,0x10,
   0x02,0x05,0x06,0x0A,0x05,0x08,0x0D,0x10,0x06,0x0D,0x0B,0x0F,0x0A,0x10,0x0F,0x13,
   0x03,0x07,0x06,0x0D,0x06,0x0D,0x0B,0x0F,0x07,0x0C,0x0D,0x0E,0x0D,0x0E,0x0F,0x12,
   0x05,0x09,0x0A,0x10,0x0A,0x10,0x0F,0x13,0x0D,0x0E,0x0F,0x12,0x0F,0x12,0x11,0x14,
   0x01,0x04,0x03,0x06,0x03,0x06,0x07,0x0D,0x02,0x06,0x05,0x0A,0x05,0x0A,0x09,0x10,
   0x03,0x06,0x07,0x0D,0x07,0x0D,0x0C,0x0E,0x06,0x0B,0x0D,0x0F,0x0D,0x0F,0x0E,0x12,
   0x02,0x06,0x05,0x0A,0x06,0x0B,0x0D,0x0F,0x05,0x0D,0x08,0x10,0x0A,0x0F,0x10,0x13,
   0x05,0x0A,0x09,0x10,0x0D,0x0F,0x0E,0x12,0x0A,0x0F,0x10,0x13,0x0F,0x11,0x12,0x14,
   0x02,0x06,0x06,0x0B,0x05,0x0A,0x0D,0x0F,0x05,0x0D,0x0A,0x0F,0x08,0x10,0x10,0x13,
   0x05,0x0A,0x0D,0x0F,0x09,0x10,0x0E,0x12,0x0A,0x0F,0x0F,0x11,0x10,0x13,0x12,0x14,
   0x05,0x0D,0x0A,0x0F,0x0A,0x0F,0x0F,0x11,0x09,0x0E,0x10,0x12,0x10,0x12,0x13,0x14,
   0x08,0x10,0x10,0x13,0x10,0x13,0x12,0x14,0x10,0x12,0x13,0x14,0x13,0x14,0x14,0x15
};



/***/

double symK (const size_t rBPFD[CELL8_SYMM_GROUPS])
{
   I64 k=0;
   for (int i= 0; i<CELL8_SYMM_GROUPS; i++) { k+= gWK[i] * (signed)rBPFD[i]; }
   return( (double) k * M_PI / 6 );
} // symK

static void addZMapU8 (size_t r[], const size_t vA[], const int nA, const uint8_t map[])
{
// size_t s=0;
   for (int i= 0; i<nA; i++)
   {
      r[ map[i] ]+= vA[i]; // s+= vA[i];
   }
} // addZMapU8


/***/

void mkfRefMeasureBPFD (float m[4], const size_t aBPFD[MKF_BINS], const float s)
{
#ifdef NO_REF_MEASURES
   m[0]= m[1]= m[2]= m[3]= -1;
#else
   refMeasures(m,aBPFD,s);
#endif
} // mkfRefMeasureBPFD

//const char *mkfGetRefML (void) { return MKF_REFML; }

int mkfSelectMeasuresBPFD (float m[4], char symCh[12], const size_t aBPFD[MKF_BINS], const float s, const int id)
{
#ifndef NO_REF_MEASURES
   if (id <= 4) { return selectRefMeasures(m, symCh, aBPFD, s, id);}
   //else...
#endif
   {
      size_t t, rBPFD[CELL8_SYMM_GROUPS]={0,};
      t= sumNZ(aBPFD, MKF_BINS);
#ifdef MKF_TEST
      if (0x01 != c8sMap[1]) { c8sGetMap(c8sMap); } // Lazy init
      //reportBytes(0,c8sMap,CELL8_PATTERNS);
      //for (int i=0; i<CELL8_PATTERNS; i++) { printf("0x%02X,", c8sMap[i]); } printf("\n\n");
#endif
      addZMapU8(rBPFD, aBPFD, MKF_BINS, c8sMap);
      m[0]= symK(rBPFD); // symMS(m+1, rBPFD);
      m[3]= volFrac(aBPFD);
      m[1]= m[2]= -1; // not yet defined
      if ((t == sumNZ(rBPFD, CELL8_SYMM_GROUPS)) && (s > 0))
      {
         float rkV= 1.0 / (s * s * s * t); //sumNZ(rBPFD, CELL8_PATTERNS);
         m[0]*= rkV;
         //m[1]*= s;
         //m[2]*= s * s;
      }
      memcpy(symCh,"Kv       Vv",12);
      return(2);
   }
   //return(0);
} // mkfSelectMeasuresBPFD

float volFrac (const size_t aBPFD[MKF_BINS])
{
   size_t s[2]={0,0};
   for (int i= 0; i<MKF_BINS; i+= 2)
   {
      s[0]+= aBPFD[i];
      s[1]+= aBPFD[i+1];
   }
   //LOG_CALL("() - s[]={%zu, %zu} (%zu)\n", s[0], s[1], s[0]+s[1]);
   return( s[1] / (float)(s[0] + s[1]) );
} // volFrac



/***/

#ifdef MKF_TEST

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
} // volFrac8

void mkfuTest (int m)
{
   LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));
   if (m & 0x0F) { c8sTest(); }
   if (m & 0xF0) { measurePatternTest(m >> 4); }
} // mkfuTest

#endif // MKF_TEST
