// ctUtil.cu - cuda test utils NB: .cu assumes c++ style compilation
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#include "ctUtil.h"


int ctuInfo (void)
{
   cudaError_t r;
   int n=0, rv=0, dv=0;

   r= cudaRuntimeGetVersion(&rv);
   r= cudaDriverGetVersion(&dv);
   //LOG("----\ncuda runtimeV%d driverV%d\n----\n", rv, dv);

   r= cudaGetDeviceCount(&n);
   /*
   for (int i= 0; i<n; i++)
   {
      cudaDeviceProp prop;
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
   */
   return(n);
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
   //pHV= (float*) malloc(bytes);
   //r= cudaHostAlloc(&pHV, bytes, cudaHostAllocMapped);

   if ((NULL == pC->pHF) && (pC->bytesF > 0)) { r= cudaMallocHost(&(pC->pHF), pC->bytesF); sE+= (0!=r); }
   if ((NULL == pC->pHU) && (pC->bytesU > 0)) { r= cudaMallocHost(&(pC->pHU), pC->bytesU); sE+= (0!=r); }
   if ((NULL == pC->pHZ) && (pC->bytesZ > 0)) { r= cudaMallocHost(&(pC->pHZ), pC->bytesZ); sE+= (0!=r); }
   LOG("cudaMallocHost() - %d: %p %zu, %p %zu, %p %zu\n", r, pC->pHF, pC->bytesF, pC->pHU, pC->bytesU, pC->pHZ, pC->bytesZ);

   if ((NULL == pC->pDF) && (pC->bytesF > 0)) { r= cudaMalloc(&(pC->pDF), pC->bytesF); sE+= (0!=r); }
   if ((NULL == pC->pDU) && (pC->bytesU > 0)) { r= cudaMalloc(&(pC->pDU), pC->bytesU); sE+= (0!=r); }
   if ((NULL == pC->pDZ) && (pC->bytesZ > 0)) { r= cudaMalloc(&(pC->pDZ), pC->bytesZ); sE+= (0!=r); }
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

#if 0
__global__ void vAddB (float r[], const float a[], const float b[], const int n)
{
   int i= blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) { r[i]= a[i] + b[i]; }
} // vAddB

void sanityTest (Context *pC)
{
   const int n= 1024;
   int i, e=0;
   for (i=0; i<n; i++) { pC->pHF[i]= i; pC->pHF[2*n - (1+i)]= 1+i; }
   cudaMemcpy(pC->pDF, pC->pHF, 2*n*sizeof(pC->pHF[0]), cudaMemcpyHostToDevice); ctuErr(NULL, "cudaMemcpy 1");
   vAddB<<<8,128>>>(pC->pDF+2*n, pC->pDF+0, pC->pDF+n, n);
   cudaMemcpy(pC->pHF+2*n, pC->pDF+2*n, n*sizeof(pC->pHF[0]), cudaMemcpyDeviceToHost); ctuErr(NULL, "cudaMemcpy 2");

   i= 2 * n;
   LOG("sanityTest() - vAddB() - [%d]=%G", i, pC->pHF[i]);
   for ( ; i < (3*n)-1; i++)
   {
      if (pC->pHF[i] != n) { ++e; LOG(" [%d]=%G", i, pC->pHF[i]); }
   }
   LOG(" [%d]=%G\n", i, pC->pHF[i]);

   //printf("*e=%d*\n", e);
} // sanityTest();
#endif

