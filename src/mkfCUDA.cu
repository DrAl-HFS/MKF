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

__global__ void addPlane (uint rBPD[256], const uint * pPln0, const uint * pPln1, const int rowStride, const int defW, const int defH)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x; // ???
   __shared__ uint bpd[32][256]; // 32KB

   if (i < defH)
   {
      const uint * pRow[2]= { pPln0 + i*rowStride, pPln1 + i*rowStride };
      const int r= i & 0x1F;

      for (int j= 0; j<256; j++) { bpd[r][j]= 0; }

      addRowBPD(bpd[r], pRow, rowStride, defW);

      __syncthreads();

#if 1
      // transposed reduction allows coalescing ???
      for (int k= r; k < 256; k+= 32) // 32 -> blockDim.x ?
      {
         uint t= 0;
         for (int j= 0; j<32; j++) { t+= bpd[j][k]; } // bpd[0][k]+=
         atomicAdd( rBPD+k, t );
      }
#else
      // Reduce
      if (0 == (r & 0x3))
      {  // r : { 0, 4, 8, 12, 16, 10, 24, 28 } 8P 3I
         for (int k= r+1, k < (r+4); k++)
         {
            for (int j= 0; j<256; j++) { bpd[r][j]+= bpd[k][j]; }
         }

         __syncthreads();

         if (0 == (r & 0xF))
         {  // r : { 0, 16 } 2P 3I
            for (int k= r+4; k < (r+16); k+= 4)
            {
               for (int j= 0; j<256; j++) { bpd[r][j]+= bpd[k][j]; }
            }

            __syncthreads();

            if (0 == r)
            {
                for (int j= 0; j<256; j++) { atomic_add(rBPD[j], bpd[r][j] + bpd[r+16][j]) };
            }
         }
      }
#endif
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

extern "C" int mkfProcess (Context *pC, const int def[3], const BinMapF32 *pMC)
{
   cudaError_t r;

   if (pC->pHF)
   {
      //r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice);

      if (NULL == pC->pDF)
      {
         r= cudaMalloc(&(pC->pDF), pC->bytesF);
         ctuErr(&r, "cudaMalloc()");
         if (pC->pDF)
         {
            r= cudaMemcpy(pC->pDF, pC->pHF, pC->bytesF, cudaMemcpyHostToDevice);
            ctuErr(&r, "cudaMemcpy()");
         }
      }

      if (NULL == pC->pDU)
      {
         r= cudaMalloc(&(pC->pDU), pC->bytesU);
         ctuErr(&r, "cudaMalloc()");
      }
      if (NULL == pC->pDZ)
      {
         r= cudaMalloc(&(pC->pDZ), pC->bytesZ);
         ctuErr(&r, "cudaMalloc()");
      }
      if (NULL == pC->pHZ)
      {
         r= cudaMallocHost(&(pC->pHZ), pC->bytesZ);
         ctuErr(&r, "cudaMalloc()");
      }

      if (pC->pDF && pC->pDU)
      {
         int blkD= 256;
         int nBlk;

         if (pC->nF <= blkD) { blkD= BLKD; }
         nBlk= (pC->nF + blkD-1) / blkD;
         // CAVEAT! Treated as 1D
         vThresh32<<<nBlk,blkD>>>(pC->pDU, pC->pDF, pC->nF, *pMC);
         ctuErr(NULL, "vThresh32()");

         if (pC->pHU)
         {
            LOG("cudaMemcpy(%p, %p, %u)\n", pC->pHU, pC->pDU, pC->bytesU);
            r= cudaMemcpy(pC->pHU, pC->pDU, pC->bytesU, cudaMemcpyDeviceToHost);
            ctuErr(NULL, "{vThresh32+} cudaMemcpy()");
         }

         if (pC->pDZ)
         {
            //size_t bpdBytes= 256*sizeof(uint);
            //if ((pC->pDZ) && (pC->bytesZ >= bpdBytes))
            uint *pBPD= (uint*)(pC->pDZ);
            const int rowStride= def[0] / 32;
            const int nRowPairs= def[1]-1;
            const int nPlanePairs= def[2]-1;
            const int planeStride= def[1] * rowStride;

            if (nRowPairs <= blkD) { blkD= BLKD; }
            nBlk= (nRowPairs + blkD-1) / blkD;

            for (int i= 0; i < nPlanePairs; i++)
            {
               const uint *pP0= pC->pDU + i * planeStride;
               const uint *pP1= pC->pDU + (i+1) * planeStride;
               addPlane<<<nBlk,blkD>>>(pBPD, pP0, pP1, rowStride, def[0], nRowPairs);
               if (0 != ctuErr(NULL, "addPlane"))
               { LOG(" .. <<<%d,%d>>>(%p, %p, %p ..)", nRowPairs, BLKD, pBPD, pP0, pP1); }
            }
            if (pC->pHZ)
            {
               r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost);
               ctuErr(&r, "{addPlane+} cudaMemcpy()");
            }
         }
      }
   }

   return(1); //0 == r);
} // mkfProcess


#ifdef MKF_CUDA_MAIN

#include "geomHacks.h"

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

void dumpUX (const uint u[], const int n)
{
   const int wrap=16;
   int i=0;
   while (i<n)
   {
      int k= i + wrap;
      if (k > n) { k= n; }
      for (int j= i; j < k; j++) { printf("%08X ", u[j]); }
      printf("\n");
      i+= k;
   }
} // dumpUX

void mkft (Context *pC, const int def[3], const float radius)
{
   BinMapF32 bmc;
   float vr, fracR= radius / def[0];
   int n;

#if 1
   vr= sphereVol(fracR);
   n= genBall(pC->pHF, def, radius);
   LOG("ball=%zu (/%d=%G, ref=%G)\n", n, pC->nF, (F64)n / pC->nF, vr);
#else
   vr= boxVol(fracR);
   n= genBlock(pC->pHF, def, radius);
   LOG("block=%zu (/%d=%G, ref=%G)\n", n, pC->nF, (F64)n / pC->nF, vr);
#endif
   setBinMapF32(&bmc,">=",0.5);
//   vf= volFrac(aBPD);
//   kf= chiEP3(aBPD);
//   LOG("volFrac=%G (ref=%G)\n", vf, vr);
//   LOG("chiEP=%G (ref=%G)\n", kf, 4 * M_PI);
   LOG("%smkfProcess() - %s","***\n","\n");
   mkfProcess(pC, def, &bmc);

   LOG("%p[%u]:\n",pC->pHU,pC->nU);
   dumpUX(pC->pHU, pC->nU);
} // mkft

__global__ void vAddB (float r[], const float a[], const float b[], const int n)
{
   int i= blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n) { r[i]= a[i] + b[i]; }
} // vAddB

int main (int argc, char *argv[])
{
   const int def[3]= {128,128,128};
   Context cux={0};

   if (buffAlloc(&cux, def, 0))
   {
#if 1
      const int n= 1024;
      for (int i=0; i<n; i++) { cux.pHF[i]= i; cux.pHF[2*n - (1+i)]= 1+i; }
      cudaMemcpy(cux.pDF, cux.pHF, 2*n*sizeof(cux.pHF[0]), cudaMemcpyHostToDevice); ctuErr(NULL, "cudaMemcpy 1");
      vAddB<<<8,128>>>(cux.pDF+2*n, cux.pDF+0, cux.pDF+n, n);
      cudaMemcpy(cux.pHF+2*n, cux.pDF+2*n, n*sizeof(cux.pHF[0]), cudaMemcpyDeviceToHost); ctuErr(NULL, "cudaMemcpy 2");
      {
         int i= 2 * n;
         LOG("vAddB() - [%d]=%G", i, cux.pHF[i]);
         for ( ; i < (3*n)-1; i++)
         {
            if (cux.pHF[i] != n) { LOG(" [%d]=%G", i, cux.pHF[i]); }
         }
         LOG(" [%d]=%G\n", i, cux.pHF[i]);
      }
      printf("***\n");
#endif
      mkft(&cux, def, 0.5*def[0] - 1.5);
      cuBuffRelease(&cux);
      cudaDeviceReset();
   }
} // main


#endif // MKF_CUDA_MAIN
