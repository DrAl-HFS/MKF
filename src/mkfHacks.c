// mkfHacks.c - dumping ground for junk experiments.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#include "mkfHacks.h"

#include "weighting.inc"

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

/***/

double sumProdZxF64 (const size_t z[], const double w[], const int n)
{
   double s= z[0] * w[0];
   for (int i= 1; i<n; i++) { s+= z[i] * w[i]; }
   return(s);
} // sumProdZxF64

void symMS (float m[2], const size_t rBPFD[CELL8_SYMM_GROUPS])
{
   m[1]= sumProdZxF64(rBPFD, gWM, CELL8_SYMM_GROUPS);
   m[2]= sumProdZxF64(rBPFD, gWS, CELL8_SYMM_GROUPS);
} // symMS

/*
float chiEP3 (const size_t aBPFD[MKF_BINS])
{
   I64 k=0;
   for (int i= 0; i<MKF_BINS; i++) { k+= (I64)gWEP3[i] * (signed)aBPFD[i]; }
   //LOG_CALL("() - k[]={%i, %i}\n", k[0], k[1]);
   return( (float) k * M_PI / 6 );
} // chiEP3
*/

static void verifyGroupK (const uint8_t patBuf[CELL8_PATTERNS], const GroupInf inf[CELL8_SYMM_GROUPS])
{
   int8_t wK[CELL8_SYMM_GROUPS];
   int j= 0, k= 0;

   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++)
   {
      const int n= inf[iG].count;
      LOG("\nG[%d] n=%d: ", iG, n);
      wK[iG]= gWEP3[ patBuf[j] ];
      for (int i=0; i<n; i++)
      {
         k= patBuf[j+i]; LOG("%d ", gWEP3[k]);
      }
      j+= n;
   }
   LOG("%s", "\n\n");
   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++) { LOG("%d, ", wK[iG]); }
} // verifyGroupK

extern void refMeasures (float m[4], const size_t aBPFD[], const float s);

static void logWeightSM (const uint8_t patBuf[CELL8_PATTERNS], const GroupInf inf[CELL8_SYMM_GROUPS])
{
   size_t d[CELL8_PATTERNS]= {0,};
   float s[CELL8_GROUP_MAX];
   float m[CELL8_GROUP_MAX];
   float r[4];
   int j= 0;

   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++)
   {
      const int n= inf[iG].count;
      LOG("\n-\nG[%d] n=%d", iG, n);
      for (int i=0; i<n; i++)
      {
         int id= patBuf[j+i];
         d[id]= 1;
         refMeasures(r,d,1.0);
         s[i]= r[2];
         m[i]= r[1];
         d[id]= 0;
      }
      j+= n;
      LOG("\n%s", "wS[]= ");
      for (int i=0; i < n; i++) { LOG("%G, ", s[i]); }
      LOG("\n%s", "wM[]= ");
      for (int i=0; i < n; i++) { LOG("%G, ", m[i]); }
   }
/*   LOG("%s", "\n\nwS[]= ");
   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++) { LOG("%G, ", s[iG]); }
   LOG("%s", "\n\nwM[]= ");
   for (int iG=0; iG < CELL8_SYMM_GROUPS; iG++) { LOG("%G, ", m[iG]); } */
} // logWeightSM

void measurePatternTest (int m)
{
   uint8_t patBuf[CELL8_PATTERNS];
   GroupInf inf[CELL8_SYMM_GROUPS];

   LOG_CALL("() [FP=%p] GroupInf=%d\n",__builtin_frame_address(0), sizeof(GroupInf));
   if (CELL8_SYMM_GROUPS == c8sGetPattern(patBuf, inf))
   {
      if (m & 0x01) { verifyGroupK(patBuf, inf); }
      if (m & 0x02) { logWeightSM(patBuf, inf); }
   }

   LOG("%s", "\n***\n");
} // measurePatternTest
