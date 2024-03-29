// mkfTest.c - main() entry-point for test code.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Oct 2019

#include "mkfACC.h"
#include "mkfCUDA.h"
#include "geomHacks.h"

/***/


B32 buffAlloc (Context *pC, const int def[3], const NumEnc e, const int n)
{
   int b, r=0;

   pC->nElem=  prodNI(def,3);
   pC->nField= MAX(1,n);
   pC->bytesF= encSizeN(&b, pC->nElem * pC->nField, e);
   if (b <= 0) { return(FALSE); }
   //else
   pC->enc= e;
   pC->bytesElem= BITS_TO_BYTES(b);
   pC->nU= setBMO(&(pC->bmo), def, 0);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= MKF_BINS;
   pC->bytesZ= sizeof(size_t) * pC->nZ; // void * sizeof(*(pC->pHZ))

   LOG("F: %d * %d * %d -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nElem, pC->nField, pC->bytesElem, pC->bytesF, pC->nU, pC->bytesU);

#ifdef MKF_CUDA
   if (cuBuffAlloc(pC,0)) { r= 2; }
#else
   pC->pHF= malloc(pC->bytesF);
   pC->pHU= malloc(pC->bytesU);
#endif
   if (pC->pHF) { memset(pC->pHF, 0, pC->bytesF); ++r; }
   if (pC->pHU) { memset(pC->pHU, 0xFF, pC->bytesU); ++r; }
   LOG("pHF=%p pHU=%p\n", pC->pHF, pC->pHU);
   return(r >= 2);
} // buffAlloc

void buffRelease (Context *pC)
{
#ifdef MKF_CUDA
   cuBuffRelease(pC);
#else
   if (NULL != pC->pHF) { free(pC->pHF); pC->pHF= NULL; }
   if (NULL != pC->pHU) { free(pC->pHU); pC->pHU= NULL; }
#endif
} // buffRelease

void checkNZ (const size_t u[], const int n, const char *pVrbFmt)
{ // debug...
   size_t t[2]= { 0, 0 };
   for (int i= 0; i < n; i++)
   {
      if (pVrbFmt && (0 != u[i])) { LOG(pVrbFmt, i, u[i]); }
      t[0]+= u[i] * bitCountZ(i);
      t[1]+= bitCountZ(u[i]);
   }
   LOG_CALL("(.. %d ..) - bitcounts: dist=%zu /8= %zu, raw=%zu\n", n, t[0], t[0]>>3, t[1]);
} // checkNZ

int compareNZ (const size_t u0[], const size_t u1[], const int n, const int flags)
{
   int d, sDiff=0, nDiff= 0;
   size_t s[2];
   s[0]= sumNZ(u0,n); s[1]= sumNZ(u1, n);
   if (s[0] != s[1])
   {
      LOG("compareNZ() - s[]= %zu, %zu (ratio: %G %G)\n", s[0], s[1], (float)s[0] / s[1], (float)s[1]/ s[0]);
   }
   else
   {
      for (int i= 0; i < n; i++)
      {
         d= (int)(u1[i] - u0[i]);
         sDiff+= d;
         nDiff+= 0 != d;
         if (flags && d) { LOG("[0x%X] %zu %zu\n", i, u0[i], u1[i]); }
      }
      LOG("compareNZ() - sDiff=%d, nDiff=%d\n", sDiff, nDiff);
   }
   return(nDiff);
} // compareNZ

void reportMeasures (const size_t a[256], const float mScale, const SMVal dts)
{
   float m[4];
   char s[16]; s[13]= 0;
   mkfSelectMeasuresBPFD(m, s, a, mScale, 2);
   LOG(" %s: %G %G %G %G\t(%Gsec)\n", s, m[0], m[1], m[2], m[3], dts);
} // reportMeasures

size_t bitCountNZ (size_t z[], const int n)
{  size_t s= bitCountZ(z[0]);
   for (int i=1; i<n; i++) { s+= bitCountZ(z[i]); }
   return(s);
} // bitCountNZ

int main (int argc, char *argv[])
{
   int id=3, def[3]= { 256, 256, 256 }; // ensure def[0] is "irregular" on first invocation or OpenACC has problems (caching?)
   BinMapF64 bmc;
   size_t aBPFD1[MKF_BINS]={0,}, aBPFD2[MKF_BINS]={0,}, *pBPFD= NULL;
   const size_t *pKBPFD=NULL;
   Context cux={0};
   SMVal dt;
   //ctuInfo();
   //geomTest(2,2);
   //c8sTest();
   //mkfuTest(0);
   //{  float m[4]; mkfMeasureBPFD(m,NULL,aBPFD1,1,1); }
   //printf("long int = %dbytes\n", sizeof(long int));
   if (buffAlloc(&cux, def, ENC_F64, 1))
   {
      const float param[]= {256-64, 1, 0}; //midRangeHNI(def,3)-3;
      const float mScale= 3.0 / sumNI(def,3); // reciprocal mean
      float vfR= genPattern(NULL, cux.pHF, def, cux.enc, cux.nField, id, param);

      if (vfR <= 0) { WARN("genPattern() - vfR=%G\n", vfR); }

      setBinMapF64(&bmc,">=",0.5);
      setupAcc(0,1);
      pBPFD= aBPFD1;
      if (ENC_F64 == cux.enc)
      {
         LOG("***\nmkfAccGetBPFDSimple(%p) - \n", pBPFD);
         deltaT();
         mkfAccGetBPFDSimple(pBPFD, cux.pHU, cux.pHF, def, &bmc);
         //LOG("bitCountNU32() - %zu\n", bitCountNU32(cux.pHU, cux.bytesU / sizeof(U32)));
         dt= deltaT();
         pKBPFD= pBPFD;
         reportMeasures(pBPFD, mScale, dt);
      }

#ifdef MKF_CUDA
      pBPFD= cux.pHZ;
      LOG("***\nMKF_CUDA: mkfCUDAGetBPFDautoCtx(%p) - \n", pBPFD);
      deltaT();
      if (mkfCUDAGetBPFDautoCtx(&cux, def, &bmc, 0x00))
      {
         dt= deltaT();
         reportMeasures(pBPFD, mScale, dt);
         if (pKBPFD) { compareNZ(pBPFD, pKBPFD, MKF_BINS, 1); }
      } else { cudaDeviceReset(); }
#endif // MKF_CUDA

#ifdef MKF_ACC_CUDA_INTEROP
      pBPFD= aBPFD2;
      LOG("***\nMKF_ACC_CUDA_INTEROP: mkfAccCUDAGetBPFD(%p) - \n", pBPFD);
      deltaT();
      if (mkfAccCUDAGetBPFD(pBPFD, cux.pHF, def, cux.enc, &bmc))
      {
         dt= deltaT();
         reportMeasures(pBPFD, mScale, dt);
         if (pKBPFD) { compareNZ(pBPFD, pKBPFD, MKF_BINS, 1); }
      } else { cudaDeviceReset(); }
#endif // MKF_ACC_CUDA_INTEROP

      if (pKBPFD && (def[0] != def[2]))
      {
         pBPFD= aBPFD2;
         LOG("***\nSWAP() - mkfAccGetBPFDSimple(%p) - \n", pBPFD);
         SWAP(int,def[0],def[2]);
         vfR= genPattern(NULL, cux.pHF, def, cux.enc, cux.nField, id, param);
         deltaT();
         mkfAccGetBPFDSimple(pBPFD, cux.pHU, cux.pHF, def, &bmc);
         dt= deltaT();
         reportMeasures(pBPFD, mScale, dt);
         compareNZ(pKBPFD, pBPFD, MKF_BINS, 0x0);
      }
   }
   buffRelease(&cux);
	return(0);
} // main
