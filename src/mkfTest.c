
//#include "mkfTools.h"
#include "binMapAcc.h"
#include "mkfAcc.h"
#include "mkfCUDA.h"
#include "mkfUtil.h"
#include "geomHacks.h"


/***/

Bool32 buffAlloc (Context *pC, const int def[3])
{
   int r=0, vol= def[0] * def[1] * def[2];

   pC->nF= vol;
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= BITS_TO_WRDSH(vol,5);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= 256;
   pC->bytesZ= 8 * pC->nZ; // void * sizeof(*(pC->pHZ))

   LOG("F: %zu -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nF, pC->bytesF, pC->nU, pC->bytesU);

#ifdef MK_CUDA
   if (cuBuffAlloc(pC,vol)) { r= 2; }
#else
   pC->pHF= malloc(pC->bytesF);
   pC->pHU= malloc(pC->bytesU);
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

void checkNU32 (const U32 u[], const int n, const char *pVrbFmt)
{ // debug...
   size_t t[2]= { 0, 0 };
   for (int i= 0; i < n; i++)
   {
      if (pVrbFmt && (0 != u[i])) { LOG(pVrbFmt, i, u[i]); }
      t[0]+= u[i] * bitCountZ(i);
      t[1]+= bitCountZ(u[i]);
   }
   LOG("checkNU32(.. %d ..) - bitcounts: dist=%zu /8= %zu, raw=%zu\n", n, t[0], t[0]>>3, t[1]);
} // checkNU32

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
      n= genBlock(cux.pHF, def, radius);
      LOG("block=%zu (/%d=%G)\n", n, cux.nF, (F64)n / cux.nF);
#endif
      setBinMapF32(&bmc,">=",0.5);
      procSimple(aBPD, cux.pHU, cux.pHF, def, &bmc);

      checkNU32(aBPD, 256, NULL);

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
         checkNU32(pBPD, 256, "%d: %u\n");
         checkNU32(cux.pHU, cux.nU, NULL); // "[%d]: 0x%04X\n"
      }
#endif
   }
   buffRelease(&cux);
	return(0);
} // main
