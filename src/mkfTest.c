
#include "mkfTools.h"
#include "binMap.h"
#include "mkf.h"

/*  */

// -> GEOMETRY ?
F32 sphereVol (F32 r) { return((4 * M_PI / 3) * r * r *r); }

F32 boxVol (F32 r) { return(8 * r * r * r); }

F32 mag2 (F32 dx, F32 dy, F32 dz) { return(dx*dx + dy*dy + dz*dz); }

/***/

int genBall (F32 *pF, const int def[3], const F32 r)
{
   size_t i= 0, n= 0;
   F32 c[3], r2= r * r;

   for (int d=0; d<3; d++) { c[d]= 0.5 * def[d]; }

   for (int j= 0; j<def[2]; j++)
   {
      for (int k= 0; k<def[1]; k++)
      {
         for (int l= 0; l<def[0]; l++)
         {
            if (mag2(j-c[0], k-c[1], l-c[2]) <= r2) { pF[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBall

int genBox (F32 *pF, const int def[3], const F32 r)
{
   size_t i= 0, n= 0;
   F32 c[3];

   for (int d=0; d<3; d++) { c[d]= 0.5 * def[d]; }

   for (int j= 0; j<def[2]; j++)
   {
      for (int k= 0; k<def[1]; k++)
      {
         for (int l= 0; l<def[0]; l++)
         {
            if ( (abs(j-c[2]) <= r) && (abs(k-c[1]) <= r) && (abs(l-c[0]) <= r) ) { pF[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBox

extern int mkfProcess (Context *pC, const int def[3], const MKBMapF32 *pMC);

Bool32 buffAlloc (Context *pC, const int def[3])
{
   int r=0, vol= def[0] * def[1] * def[2];

   pC->nF= vol;
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= BITS_TO_WRDSH(vol,5);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= 256;
   pC->bytesZ= 8 * pC->nZ; // void * sizeof(*(pC->pHZ))

   LOG("vol= %zu sites\nF: %zu Bytes\n", vol, pC->bytesF);

#ifdef MK_CUDA
   if (cuBuffAlloc(pC,vol)) { r= 2; }
#else
   cux.pHF= malloc(n);
   cux.pHU= malloc(n);
#endif
   if (pC->pHF) { memset(pC->pHF, 0, pC->bytesF); ++r; }
   if (pC->pHU) { memset(pC->pHU, 0xFF, pC->bytesU); ++r; }
   return(r >= 2);
} // buffAlloc

void buffRelease (Context *pC)
{
#ifdef MK_CUDA
   cuBuffRelease(pC);
#else
   if (NULL != pC->pHF) { free(pC->pHF); pC->pHF= NULL; }
   if (NULL != pC->pHU) { free(pC->pHU); pC->pHU= NULL; }
#endif
} // buffRelease

void checkBPD (const U32 aBPD[256], int verbose)
{ // debug...
   size_t t= 0;
   for (int i= 0; i < 256; i++)
   {
      if (verbose && (0 != aBPD[i])) { LOG("%d: %u\n", i, aBPD[i]); }
      t+= aBPD[i] * bitCountZ(i);
   }
   LOG("checkBPD() - bitcount=%zu /8= %zu\n", t, t>>3);
} // checkBPD

int main (int argc, char *argv[])
{
   const int def[3]= {64,64,64};
   const F32 radius= 0.5*def[0] - 1.5;
   F32 fracR;
   BinMapF32 bmc;
   U32 aBPD[256]={0,};
   MKMeasureVal vf, vr, kf;
   Context cux={0};
   int n;

   if (buffAlloc(&cux,def))
   {
      fracR= radius / def[0];
#if 1
      vr= sphereVol(fracR);
      n= genBall(cux.pHF, def, radius);
      LOG("ball=%zu (/%d=%G)\n", n, cux.nF, (F64)n / cux.nF);
#else
      vr= boxVol(fracR);
      n= genBox(cux.pHF, def, radius);
      LOG("box=%zu (/%d=%G)\n", n, cux.nF, (F64)n / cux.nF);
#endif
      setBMCF32(&bmc,">=",0.5);
      procSimple(aBPD, cux.pHU, cux.pHF, def, &bmc);

      checkBPD(aBPD, 0);

      vf= volFrac(aBPD);
      kf= chiEP3(aBPD);
      LOG("volFrac=%G (ref=%G)\n", vf, vr);
      LOG("chiEP=%G (ref=%G)\n", kf, 4 * M_PI);
#ifdef MK_CUDA
      LOG("%smkfProcess() - %s","***\n","\n");
      if (mkfProcess(&cux, def, &bmc))
      {
         const uint *pBPD= cux.pHZ;
         LOG("\tvolFrac=%G chiEP=%G\n", volFrac(pBPD), chiEP3(pBPD));
         checkBPD(pBPD, 1);
      }
#endif
   }
   buffRelease(&cux);
   //utilTest();
	//LOG_CALL("() %s\n", "MKF");
	return(0);
} // main
