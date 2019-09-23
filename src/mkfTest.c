
#include "mkfACC.h"
#include "mkfCUDA.h"
#include "geomHacks.h"
#ifdef __PGI
#include <openacc.h> // -> /opt/pgi/linux86-64/2019/include/openacc.h
// PGI: _DEF_OPENACC
// GNU: _OPENACC_H
#define OPEN_ACC_API
#endif

/***/

int setupAcc (int id)
{  // Only multicore acceleration works presently: GPU produces garbage...
   int r=-1;
#ifdef OPEN_ACC_API
   if (1 == id) { id= acc_device_nvidia; } else { id= acc_device_host; }
   acc_set_device_type( id );
   r= acc_get_device_type();
   LOG_CALL("() - acc_set_device_type( * ) - 0x%X -> %d\n", "", id, r);
#endif
   return(r);
} // setupAcc

B32 buffAlloc (Context *pC, const int def[3])
{
   //const int nC= prodOffsetNI(def, 3, -1);
   //const int lines= def[1]*def[2];
   int r=0;

   pC->nF= prodOffsetNI(def,3,0);
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= setBMO(&(pC->bmo), def, 0);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= MKF_BINS;
   pC->bytesZ= sizeof(size_t) * pC->nZ; // void * sizeof(*(pC->pHZ))

   LOG("F: %zu -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nF, pC->bytesF, pC->nU, pC->bytesU);

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

void reportMeasures (const size_t a[256], const float mScale)
{
   float m[4];
   if (mkfMeasureBPFD(m, a, mScale, 0))
   {
      LOG(" K M S V: %G %G %G %G\n", m[0], m[1], m[2], m[3]);
   }
} // reportMeasures

size_t bitCountNZ (size_t z[], const int n)
{  size_t s= bitCountZ(z[0]);
   for (int i=1; i<n; i++) { s+= bitCountZ(z[i]); }
   return(s);
} // bitCountNZ

int main (int argc, char *argv[])
{
   int id=3, def[3]= {256, 256, 256}; // ensure def[0] is "irregular" on first invocation or OpenACC has problems (caching?)
   BinMapF32 bmc;
   size_t *pBPFD=NULL, aBPFD1[MKF_BINS]={0,}, aBPFD2[MKF_BINS]={0,};
   Context cux={0};

   //ctuInfo();
   //geomTest(2,2);
   //c8sTest();
   mkfuTest(0);
   //printf("long int = %dbytes\n", sizeof(long int));
   if (buffAlloc(&cux,def))
   {
      const float param= 256-64; //midRangeHNI(def,3)-3;
      const float mScale= 3.0 / sumNI(def,3); // reciprocal mean
      float vfR= genPattern(cux.pHF, def, 32, id, param);

      if (vfR <= 0) { WARN("genPattern() - vfR=%G\n", vfR); }

      setBinMapF32(&bmc,">=",0.5);
      setupAcc(0);
      LOG("***\nmkfAccGetBPFDSimple(%p) - \n", aBPFD1);
      mkfAccGetBPFDSimple(aBPFD1, cux.pHU, cux.pHF, def, &bmc);
      reportMeasures(aBPFD1, mScale);

      if (NULL == pBPFD) { pBPFD= cux.pHZ; }
#ifdef MKF_CUDA
      LOG("***\nMKF_CUDA: mkfCUDAGetBPFDautoCtx(%p) - \n", pBPFD);
      if (mkfCUDAGetBPFDautoCtx(&cux, def, &bmc, 0x10))
      {
         //LOG("bc=%zu\n", bitCountNZ(cux.pHU, cux.bytesU/sizeof(size_t)));
         reportMeasures(pBPFD, mScale);
         compareNZ(aBPFD1, pBPFD, MKF_BINS, 1);
      }
#endif // MKF_CUDA

#ifdef MKF_ACC_CUDA_INTEROP
      LOG("***\nMKF_ACC_CUDA_INTEROP: mkfAccCUDAGetBPFD(%p) - \n", aBPFD2);
      if (mkfAccCUDAGetBPFD(aBPFD2, cux.pHU, cux.pHF, def, &bmc))
      {
         reportMeasures(aBPFD2, mScale);
         compareNZ(aBPFD1, aBPFD2, MKF_BINS, 1);
      }
#endif // MKF_ACC_CUDA_INTEROP

      if (def[0] != def[2])
      {
         LOG("***\nSWAP() - mkfAccGetBPFDSimple(%p) - \n", aBPFD2);
         SWAP(int,def[0],def[2]);
         vfR= genPattern(cux.pHF, def, 32, id, param);
         mkfAccGetBPFDSimple(aBPFD2, cux.pHU, cux.pHF, def, &bmc);
         reportMeasures(aBPFD2, mScale);
         compareNZ(aBPFD1, aBPFD2, MKF_BINS, 0x0);
      }
   }
   buffRelease(&cux);
	return(0);
} // main
