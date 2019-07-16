// mkf.cu - Minkowski Functional pattern processing using CUDA NB: .cu assumes c++ style compilation
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#ifndef MKF_CUDA_CU
#define MKF_CUDA_CU // supress header "multiple definition" glitch
#endif

#include "mkfCUDA.h"

#ifdef MKF_CUDA_CU
#undef MKF_CUDA_CU // header glitch supression done
#endif

// Wide counter for atomicAdd (nvcc dislikes size_t)
typedef unsigned long long CUACount;


/***/

// CUDA kernels and wrappers

#define BLKS 5
#define BLKD (1<<BLKS)
#define BLKM BLKD-1
//define BLKN 1024/BLKD

__global__ void vThresh8 (uint r[], const float f[], const size_t n, const BinMapF32 mc)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[BLKD];
   if (i < n)
   {
      const int j= i & BLKM;
      const int k= i & 0x7; // j & 0x7
      const int d= 1 + (f[i] > mc.t[0]) - (f[i] < mc.t[0]);
      z[j]= ((mc.m >> d) & 0x1) << k; // smaller shift faster ?

      __syncthreads();

      if (0 == k)
      {  // j : { 0, 8, 16, 24 } 4P, 7I
         for (int l=1; l<8; l++) { z[j]|= z[j+l]; }

         __syncthreads();

         if (0 == j)
         {
            r[i>>BLKS]= ( z[0] << 0 ) | ( z[8] << 8 ) | ( z[16] << 16 ) | ( z[24] << 24 );
         }
      }
   }
} // vThresh8

__global__ void vThresh32 (uint r[], const float f[], const size_t n, const BinMapF32 mc)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[BLKD];
   if (i < n)
   {
      const int j= i & BLKM;
      const int d= 1 + (f[i] > mc.t[0]) - (f[i] < mc.t[0]);
      z[j]= ((mc.m >> d) & 0x1) << j; // assume "barrel" shifter

      __syncthreads();

      if (0 == (j & 0x3))
      {  // j : { 0, 4, 8, 12, 16, 10, 24, 28 } 8P 3I
         for (int l=1; l<4; l++) { z[j]|= z[j+l]; }

         __syncthreads();

         //if (0 == j) { r[i>>BLKS]= z[0] | z[4] | z[8] | z[12] | z[16] | z[20] | z[24] | z[28]; }
         if (0 == (j & 0xF))
         {  // j : { 0, 16 } 2P 3I
            for (int l=4; l<16; l+=4) { z[j]|= z[j+l]; }

            __syncthreads();

            if (0 == j) { r[i>>BLKS]= z[0] | z[16]; }
         }
      }
   }
} // vThresh32

#define CHUNK_SHIFT (5)
#define CHUNK_SIZE (1<<CHUNK_SHIFT)
#define CHUNK_MASK (CHUNK_SIZE-1)

__device__ void loadChunk
(
   size_t bufChunk[4],     // chunk buffer
   const uint * pR0, // Location within row of first -
   const uint * pR1, //  - and second planes
   const int rowStride,       // stride between successive rows within each plane
   const int lsh              // shift at which to append chunks (0/1)
)
{  // vect
   bufChunk[0] |= (pR0[0] << lsh);
   bufChunk[1] |= (pR0[rowStride] << lsh);
   bufChunk[2] |= (pR1[0] << lsh);
   bufChunk[3] |= (pR1[rowStride] << lsh);
} // loadChunk

__device__ unsigned char bp4x2 (size_t bufChunk[4])
{
   unsigned char r=  (bufChunk[0] & 0x3) |
               ((bufChunk[1] & 0x3) << 2) |
               ((bufChunk[2] & 0x3) << 4) |
               ((bufChunk[3] & 0x3) << 6);
   bufChunk[0] >>= 1;
   bufChunk[1] >>= 1;
   bufChunk[2] >>= 1;
   bufChunk[3] >>= 1;
   return(r);
} // bp4x2

__device__ int ap4x2xN (uint bpd[256], size_t bufChunk[4], const int n)
{
   const int m= n / 4;
   int i, j;
   for (i= 0; i < m; i++) // seq
   {
      bpd[ bp4x2(bufChunk) ]++;
      bpd[ bp4x2(bufChunk) ]++;
      bpd[ bp4x2(bufChunk) ]++;
      bpd[ bp4x2(bufChunk) ]++;
   }
   j= i * 4;
   if (j++ < n)
   {
      bpd[ bp4x2(bufChunk) ]++;
      if (j++ < n)
      {
         bpd[ bp4x2(bufChunk) ]++;
         if (j++ < n)
         {
            bpd[ bp4x2(bufChunk) ]++;
         }
      }
   }
   return(n);
} // ap4x2xN

__device__ void addRowBPD
(
   uint bpd[256], // result pattern distribution
   const uint * pRow[2],
   const int rowStride,
   const int n    // Number of single bit elements packed in row
)
{  // seq
   int m, k, i;
   size_t bufChunk[4]= { 0,0,0,0 };

   // First chunk of n bits yields n-1 patterns
   loadChunk(bufChunk, pRow[0]+0, pRow[1]+0, rowStride, 0);
   k= ap4x2xN(bpd, bufChunk, MIN(CHUNK_SIZE-1, n-1));
   // Subsequent whole chunks yield n patterns
   i= 0;
   m= n>>CHUNK_SHIFT;
   while (++i < m)
   {
      loadChunk(bufChunk, pRow[0]+i, pRow[1]+i, rowStride, 1);
      k= ap4x2xN(bpd, bufChunk, CHUNK_SIZE);
   }
   // Check for residual bits < CHUNK_SIZE
   k= n & CHUNK_MASK;
   if (k > 0)
   {
      loadChunk(bufChunk, pRow[0]+i, pRow[1]+i, rowStride, 1);
      k= ap4x2xN(bpd, bufChunk, k);
   }
} // addRowBPD

__global__ void addPlane (CUACount rBPFD[256], const uint * pPln0, const uint * pPln1, const int rowStride, const int defW, const int defH)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x; // ???
   __shared__ uint bpfd[256][BLKD]; // C-row-major (lexicographic) memory order. 32KB

   if (i < defH)
   {
      const uint * pRow[2]= { pPln0 + i*rowStride, pPln1 + i*rowStride };
      const int r= i & 0x1F;

#if 0
      for (int j= 0; j<256; j++) { bpd[j][r]= 0; }
#else // transpose zeroing for write coalescing
      for (int k= r; k < 256; k+= BLKD)
      {
         for (int j= 0; j < BLKD; j++) { bpfd[k][j]= 0; }
      }
#endif
      addRowBPD(&(bpfd[0][r]), pRow, rowStride, defW);

      __syncthreads();

      // transposed reduction should allows coalescing
      for (int k= r; k < 256; k+= BLKD)
      {
         CUACount t= 0;
         for (int j= 0; j < BLKD; j++) { t+= bpfd[k][j]; }
         atomicAdd( rBPFD+k, t );
      }
   }
} // addPlane


/***/

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

extern "C" int mkfCUDAGetBPFDSimple (Context *pC, const int def[3], const BinMapF32 *pMC)
{
   cudaError_t r;

   if (pC->pHF)
   {
      //r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice);

      if (NULL == pC->pDF)
      {
         r= cudaMalloc(&(pC->pDF), pC->bytesF);
         ctuErr(&r, "cudaMalloc()");
      }
      if (pC->pDF)
      {
         r= cudaMemcpy(pC->pDF, pC->pHF, pC->bytesF, cudaMemcpyHostToDevice);
         ctuErr(&r, "cudaMemcpy()");
      }

      if (NULL == pC->pDU)
      {
         r= cudaMalloc(&(pC->pDU), pC->bytesU);
         ctuErr(&r, "cudaMalloc()");
      }

      if (pC->pDF && pC->pDU)
      {
         int blkD= BLKD;//256;
         int nBlk;

         if (pC->pDZ) { cudaMemset(pC->pDZ, 0, pC->bytesZ); }
         if (pC->nF <= blkD) { blkD= BLKD; }
         nBlk= (pC->nF + blkD-1) / blkD;
         // CAVEAT! Treated as 1D
         vThresh32<<<nBlk,blkD>>>(pC->pDU, pC->pDF, pC->nF, *pMC);
         ctuErr(NULL, "vThresh32()");
         cudaDeviceSynchronize();

         if (pC->pHU)
         {
            LOG("cudaMemcpy(%p, %p, %u)\n", pC->pHU, pC->pDU, pC->bytesU);
            r= cudaMemcpy(pC->pHU, pC->pDU, pC->bytesU, cudaMemcpyDeviceToHost);
            ctuErr(NULL, "{vThresh32+} cudaMemcpy()");
         }

         if (pC->bytesZ> 0)
         {
            if (NULL == pC->pDZ)
            {
               r= cudaMalloc(&(pC->pDZ), pC->bytesZ);
               ctuErr(&r, "cudaMalloc()");
               cudaMemset(pC->pDZ, 0, pC->bytesZ);
            }
            if (NULL == pC->pHZ)
            {
               r= cudaMallocHost(&(pC->pHZ), pC->bytesZ);
               ctuErr(&r, "cudaMalloc()");
            }
            if (pC->pDZ)
            {
               //size_t bpdBytes= 256*sizeof(uint);
               //if ((pC->pDZ) && (pC->bytesZ >= bpdBytes))
               CUACount *pBPFD= (CUACount*)(pC->pDZ);
               const int rowStride= def[0] / 32;
               const int nRowPairs= def[1]-1;
               const int nPlanePairs= def[2]-1;
               const int planeStride= def[1] * rowStride;

               //if (nRowPairs <= blkD) {
               blkD= BLKD;
               nBlk= (nRowPairs + blkD-1) / blkD;

               for (int i= 0; i < nPlanePairs; i++)
               {
                  const uint *pP0= pC->pDU + i * planeStride;
                  const uint *pP1= pC->pDU + (i+1) * planeStride;
                  addPlane<<<nBlk,blkD>>>(pBPFD, pP0, pP1, rowStride, def[0], nRowPairs);
                  if (0 != ctuErr(NULL, "addPlane"))
                  { LOG(" .. <<<%d,%d>>>(%p, %p, %p ..)", nRowPairs, BLKD, pBPFD, pP0, pP1); }
               }
               cudaDeviceSynchronize();
               if (pC->pHZ)
               {
                  r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost);
                  ctuErr(&r, "{addPlane+} cudaMemcpy()");
               }
            }
         }
      }
   }

   return(1); //0 == r);
} // mkfCUDAGetBPFDSimple


#ifdef MKF_CUDA_MAIN

#include "geomHacks.h"
#include "mkfUtil.h"

int buffAlloc (Context *pC, const int def[3], const int blkZ)
{
   int vol= def[0] * def[1] * def[2];

   pC->nF= vol;
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= BITS_TO_WRDSH(vol,5);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= blkZ * 256;
   pC->bytesZ= 8 * pC->nZ; // void * sizeof(*(pC->pHZ))

   LOG("F: %zu -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nF, pC->bytesF, pC->nU, pC->bytesU);

   return cuBuffAlloc(pC,vol);
} // buffAlloc


static const char gSepCh[2]={' ','\n'};

void dumpF (const float f[], const int n, const int wrap)
{
   int i=0;
   while (i<n)
   {
      int k= i + wrap;
      if (k > n) { k= n; }
      for (int j= i; j < k; j++) { LOG("%G%c", f[j], gSepCh[(j+1)==k]); }
      i= k;
   }
} // dumpF

void dumpUX (const uint u[], const int n, const int wrap)
{
   int i=0;
   while (i<n)
   {
      int k= i + wrap;
      if (k > n) { k= n; }
      for (int j= i; j < k; j++) { LOG("%08X%c", u[j], gSepCh[(j+1)==k]); }
      i= k;
   }
} // dumpUX

void mkft (Context *pC, const int def[3], const float radius)
{
   BinMapF32 bmc;
   float vr, fracR= radius / def[0];
   int m, n;

#if 1
   vr= sphereVol(fracR);
   n= genBall(pC->pHF, def, radius);
   LOG("ball=%zu (/%d=%G, ref=%G)\n", n, pC->nF, (F64)n / pC->nF, vr);
#else
   vr= boxVol(fracR);
   n= genBlock(pC->pHF, def, radius);
   LOG("block=%zu (/%d=%G, ref=%G)\n", n, pC->nF, (F64)n / pC->nF, vr);
#endif
   m= def[0] * def[1] * def[2];
   //dumpF(pC->pHF+n, n, def[0]);
   setBinMapF32(&bmc,">=",0.5);
   LOG("***\nmkfCUDAGetBPFDSimple() - bmc: %f,0x%X\n",bmc.t[0], bmc.m);
   mkfCUDAGetBPFDSimple(pC, def, &bmc);
#if 0
   LOG("%p[%u]:\n",pC->pHU,pC->nU);
   m= def[0] >> BLKS; // def[0] / BLKD;
   n= m * def[1];
   if (n > pC->nU/2) { n= pC->nU/2; }
   while ((m<<1) < 16) { m<<= 1; }
   dumpUX(pC->pHU+2*n, n, m);
   LOG("%s\n","-");
   dumpUX(pC->pHU+3*n, n, m);
#endif
   if (pC->pHZ)
   {
      const size_t *pU= (size_t*)pC->pHZ;

      m= 0;
      for (int i= 0; i<256; i++)
      {
         m+= pU[i];
         if (pU[i] > 0) { LOG("[%d]=%u\n", i, pU[i]); }
      }
      LOG("sum=%u (%u)\n", m, pC->nF);
#if 1
      float vf= volFrac(pU);
      float kf= chiEP3(pU);

      LOG("volFrac=%G (ref=%G)\n", vf, vr);
      LOG("chiEP=%G (ref=%G)\n", kf, 4 * M_PI);
#endif
   }
} // mkft

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

   printf("*e=%d*\n", e);
} // sanityTest();
#endif

int main (int argc, char *argv[])
{
   const int def[3]= {64,64,64};
   Context cux={0};

   if (buffAlloc(&cux, def, 1))
   {
      //sanityTest(&cux);
      mkft(&cux, def, 0.5*def[0] - 1.5);
      cuBuffRelease(&cux);
   }
   cudaDeviceReset();
} // main

#endif // MKF_CUDA_MAIN
