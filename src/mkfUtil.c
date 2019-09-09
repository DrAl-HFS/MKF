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

static const int8_t gWK[CELL8_SYMM_GROUPS]=
{  0, 3, 0, 6, 6, -3, 3,
   9, 0, -6, -6, 0, 12, 0,
   -3, -8, -3, -12, -6, 0, 3,
   0
};
static const double gWM[CELL8_SYMM_GROUPS]=
{  0, 0.763019, 0.642656, 0.616236, 0.686801, 0.187756, 0.167269,
   0.161335, 0, -0.431447, -0.287632, 0, 0, -0.143816,
   -0.592783, -0.540019, -0.475387, -0.686801, -0.616236, -0.727774, -0.331571,
   0
};
static const double gWS[CELL8_SYMM_GROUPS]=
{  0, 0.944412, 1.05178, 0.921095, 0.496906, 0.949947, 0.865898,
   0.793181, 0.926631, 0.638921, 0.764068, 0.784616, 0.874463, 0.868666,
   0.557639, 0.682786, 0.923863, 0.496906, 0.737983, 0.65947, 0.0812817,
   0
};
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
   for (int i= 0; i<CELL8_SYMM_GROUPS; i++) { k+= gWK[i] * (signed)rBPFD[i]; }
   return( (float) k * M_PI / 6 );
} // tc

double sumProdZxF64 (const size_t z[], const double w[], const int n)
{
   double s= z[0] * w[0];
   for (int i= 1; i<n; i++) { s+= z[i] * w[i]; }
   return(s);
} // sumProdZxF64


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
         size_t t, rBPFD[CELL8_SYMM_GROUPS]={0,};
         t= sumNZ(aBPFD, MKF_BINS);
         if (0x01 != c8sMap[1]) { c8sGetMap(c8sMap); } // Lazy init
         addZMapU8(rBPFD, aBPFD, MKF_BINS, c8sMap);
         m[0]= tc(rBPFD);
         m[1]= sumProdZxF64(rBPFD, gWM, CELL8_SYMM_GROUPS) / t;
         m[2]= sumProdZxF64(rBPFD, gWS, CELL8_SYMM_GROUPS) / t;
         m[3]= -1;
         if ((t == sumNZ(rBPFD, CELL8_SYMM_GROUPS)) && (s > 0))
         {
            float rkV= 1.0 / (s * s * s * t); //sumNZ(rBPFD, CELL8_PATTERNS);
            m[0]*= rkV;
            //m[1]*= s;
            //m[2]*= s * s;
         }
         //m[1]= m[2]= m[3]= -1;
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

static void verifyGroups (const uint8_t patBuf[CELL8_PATTERNS], const GroupInf inf[CELL8_SYMM_GROUPS])
{
   int8_t wK[CELL8_SYMM_GROUPS];
   int j= 0, k= 0;

   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++)
   {
      const int n= inf[iG].count;
      LOG("\nG[%d] n=%d: ", iG, n);
      wK[iG]= gWEP3[ patBuf[j] ];
      for (int i=0; i<n; i++) { k= patBuf[j+i]; LOG("%d ", gWEP3[k]); }
      j+= n;
   }
   LOG("%s", "\n\n");
   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++) { LOG("%d, ", wK[iG]); }
} // verifyGroups

static void logMeasureW (const uint8_t patBuf[CELL8_PATTERNS], const GroupInf inf[CELL8_SYMM_GROUPS])
{
   long int d[MKF_BINS]= {0,};
   float s[CELL8_SYMM_GROUPS];
   float m[CELL8_SYMM_GROUPS];
   double res[3]={1,1,1};
   int j= 0;

   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++)
   {
      int id= patBuf[j];
      d[id]= 1;
      s[iG]= specsurf(d, res);
      m[iG]= specimc(d, res);
      d[id]= 0;
      j+= inf[iG].count;
   }
   LOG("%s", "\n\nwS[]= ");
   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++) { LOG("%G, ", s[iG]); }
   LOG("%s", "\n\nwM[]= ");
   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++) { LOG("%G, ", m[iG]); }
} // logMeasureW

static void measurePatternTest (void)
{
   uint8_t patBuf[CELL8_PATTERNS];
   GroupInf inf[CELL8_SYMM_GROUPS];

   LOG_CALL("() [FP=%p] GroupInf=%d\n",__builtin_frame_address(0), sizeof(GroupInf));
   if (CELL8_SYMM_GROUPS == c8sGetPattern(patBuf, inf))
   {
      //verifyGroups(patBuf, inf);
      //logMeasureW(patBuf, inf);
   }

   LOG("%s", "\n***\n");
} // measurePatternTest

void mkfuTest (void)
{
   LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));
   //c8sTest();
   measurePatternTest();
} // mkfuTest
