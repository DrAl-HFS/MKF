// ctUtil.cu - cuda test utils NB: .cu assumes c++ style compilation
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#include "ctUtil.h"

void ctuInfo (void)
{
   cudaError_t r;
   cudaDeviceProp prop;
   int n=0, rv=0, dv=0;

   r= cudaRuntimeGetVersion(&rv);
   r= cudaDriverGetVersion(&dv);
   LOG("----\ncuda runtimeV%d driverV%d\n----\n", rv, dv);

   r= cudaGetDeviceCount(&n);
   for (int i= 0; i<n; i++)
   {
      r= cudaGetDeviceProperties(&prop,i);
      if (0 == r)
      {
         char c[4]="   ";
         float f[3];
         LOG("Dev.  : %d/%d= %s CC%d.%d\n", i, n, prop.name, prop.major, prop.minor);
         f[0]= binSizeZ(c+0, prop.totalGlobalMem);
         f[1]= binSizeZ(c+1, prop.totalConstMem);
         f[2]= binSizeZ(c+2, prop.sharedMemPerBlock);
         LOG("Mem.  : %.3G%c C=%.3G%c S=%.3G%c\n", f[0], c[0], f[1], c[1], f[2], c[2]);
         LOG("Proc. : M=%d W=%d\n", prop.multiProcessorCount, prop.warpSize);
         LOG("Blk.  : R=%d T=%d\n", prop.regsPerBlock, prop.maxThreadsPerBlock);
         for (int i=0; i<3; i++) { f[i]= binSizeZ(c+i, prop.maxGridSize[i]); }
         LOG("Grd.  : %.3G%c %.3G%c %.3G%c\n", f[0], c[0], f[1], c[1], f[2], c[2]);
         LOG("Thrd. : %d %d %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
         LOG("Map=%d Overlap=%d\n----\n", prop.canMapHostMemory, prop.deviceOverlap);
      }
   }
} // ctuInfo

/*
#include <nvml.h>
{
   nvmlReturn_t r;
   info();
   r= nvmlInit();
   if (r) {;}
   nvmlReturn_t r= nvmlShutdown();
   if (r) {;}
}
*/

extern "C" int cuBuffAlloc (Context *pC, uint n)
{
   cudaError_t r;
   int sE=0;
   //cudaSetDevice, cudaChooseDevice
#if 0
   pC->nF= n;
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= BITS_TO_WRDSH(n,5);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= 256;
   pC->bytesZ= 8 * pC->nZ; // void * sizeof(*(pC->pHZ))
#endif
   //pHV= (float*) malloc(bytes);
   //r= cudaHostAlloc(&pHV, bytes, cudaHostAllocMapped);

   r= cudaMallocHost(&(pC->pHF), pC->bytesF); sE+= (0!=r);
   r= cudaMallocHost(&(pC->pHU), pC->bytesU); sE+= (0!=r);
   if (pC->bytesZ > 0) { r= cudaMallocHost(&(pC->pHZ), pC->bytesZ); sE+= (0!=r); }
   LOG("cudaMallocHost() - %d: %p %zu, %p %zu, %p %zu\n", r, pC->pHF, pC->bytesF, pC->pHU, pC->bytesU, pC->pHZ, pC->bytesZ);

   r= cudaMalloc(&(pC->pDF), pC->bytesF); sE+= (0!=r);
   r= cudaMalloc(&(pC->pDU), pC->bytesU); sE+= (0!=r);
   if (pC->bytesZ > 0) { r= cudaMalloc(&(pC->pDZ), pC->bytesZ); sE+= (0!=r); }
   LOG("cudaMalloc() : %p %p %p\n", pC->pDF, pC->pDU, pC->pDZ);

   return(0 == sE);
} // cuBuffAlloc

extern "C" int cuBuffRelease (Context *pC)
{
   cudaError_t r;
   int sE=0;

   if (pC->pDF) { r= cudaFree(pC->pDF); sE+= (0!=r); }
   if (pC->pDU) { r= cudaFree(pC->pDU); sE+= (0!=r); }
   if (pC->pDZ) { r= cudaFree(pC->pDZ); sE+= (0!=r); }

   if (pC->pHF) { r= cudaFreeHost(pC->pHF); sE+= (0!=r); }
   if (pC->pHU) { r= cudaFreeHost(pC->pHU); sE+= (0!=r); }
   if (pC->pHZ) { r= cudaFreeHost(pC->pHZ); sE+= (0!=r); }
   return(0 == sE);
} // ctuReleaseCtx

#if 1
cudaError_t ctuErr (cudaError_t *pE, const char *s)
{
   cudaError_t e;
   if (NULL == pE) { e= cudaGetLastError(); } else { e= *pE; }
   if (0 != e)
   {
      ERROR("%s - r=%d -> %s\n", s, e, cudaGetErrorName(e));
   }
   return(e);
} // ctuErr
#endif
