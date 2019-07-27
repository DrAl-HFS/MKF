// mkf.cu - Minkowski Functional pattern processing using CUDA NB: .cu assumes c++ style compilation
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#ifndef MKF_CUDA_CU
#define MKF_CUDA_CU // supress header "multiple definition" glitch
#endif

#include "mkfCUDA.h"
//include "binMapCUDA.h"

#ifdef MKF_CUDA_CU
#undef MKF_CUDA_CU // header glitch supression done
#endif

// Wide counter for atomicAdd (nvcc dislikes size_t)
typedef unsigned long long CUACount;


/***/

// CUDA kernels and wrappers

#define VT_BLKS 5
#define VT_BLKD (1<<VT_BLKS)
#define VT_BLKM (VT_BLKD-1)

__global__ void vThresh32 (uint r[], const float f[], const size_t n, const BinMapF32 bm)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[VT_BLKD];
   if (i < n)
   {
      const int j= i & VT_BLKM;
#if 1
      const int d= 1 + (f[i] > bm.t[0]) - (f[i] < bm.t[0]);
      z[j]= ((bm.m >> d) & 0x1) << j; // assume "barrel" shifter
#else
      z[j]= bm1f32(f[i],bm) << j;
#endif
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

            if (0 == j) { r[i>>VT_BLKS]= z[0] | z[16]; }
         }
      }
   }
} // vThresh32


#define PACK16

#ifdef PACK16
#define BPFD_W32_BINS (MKF_BINS/2)
//#define BPFD_BLKS 6
#else
#define BPFD_W32_BINS MKF_BINS
//#define BPFD_BLKS 5
#endif

//ifdef PACK16 #define BPFD_BLKS 6 #else#endif
#define BPFD_BLKS 5
#define BPFD_BLKD (1<<BPFD_BLKS)
#define BPFD_BLKM (BPFD_BLKD-1)

typedef unsigned long long ULL;
typedef uint   U16P; // ushort2 ???

#define CHUNK_SHIFT (5)
#define CHUNK_SIZE (1<<CHUNK_SHIFT)
#define CHUNK_MASK (CHUNK_SIZE-1)

class ChunkBuf
{
   ULL u00, u01, u10, u11;

   __device__ uint buildNext (void)
   {
      uint bp=  ( u00 & 0x3); u00 >>= 1;
      bp|= ((u01 & 0x3) << 2); u01 >>= 1;
      bp|= ((u10 & 0x3) << 4); u10 >>= 1;
      bp|= ((u11 & 0x3) << 6); u11 >>= 1;
      return(bp);
   }

public:
   __device__ ChunkBuf (const uint * __restrict__ pR0, const uint * __restrict__ pR1, const int rowStride)
   {
      u00= pR0[0];
      u01= pR0[rowStride];
      u10= pR1[0];
      u11= pR1[rowStride];
   }
   __device__ void loadSh1 (const uint * __restrict__ pR0, const uint * __restrict__ pR1, const int rowStride)
   {
      u00|= ( (ULL) pR0[0] ) << 1;
      u01|= ( (ULL) pR0[rowStride] ) << 1;
      u10|= ( (ULL) pR1[0] ) << 1;
      u11|= ( (ULL) pR1[rowStride] ) << 1;
   }
#ifndef PACK16
   __device__ void add (uint bpfd[], const int n)
   {
      for (int i= 0; i < n; i++) { bpfd[ buildNext() ]++; }
   } // add
#else
   __device__ void add (U16P bpfd[], const int n)
   {
      //const ushort2 lh[2]={ushort2(1,0),ushort2(0,1)}; //
      const U16P lh[2]={1,1<<16}; // even -> lo, odd -> hi (16b)

      for (int i= 0; i < n; i++)
      {
         const uint bp= buildNext();
         bpfd[ bp >> 1 ]+= lh[bp & 1];
      }
   } // add16P
#endif
}; // class ChunkBuf

__device__ void addRowBPFD
(
   uint         bpfd[], // result pattern distribution
   const uint  * __restrict__ pRow[2], // restrict relevant to const ?
   const int   rowStride,
   const int   n    // Number of single bit elements packed in row
)
{  // seq
   int m, k, i;
   ChunkBuf  cb(pRow[0]+0, pRow[1]+0, rowStride);
   k= MIN(CHUNK_SIZE-1, n-1); //dbg[0]+= k;
   cb.add(bpfd, k);
   // Subsequent whole chunks yield n patterns
   i= 0;
   m= n>>CHUNK_SHIFT;
   while (++i < m)
   {
      cb.loadSh1(pRow[0]+i, pRow[1]+i, rowStride);
      cb.add(bpfd, CHUNK_SIZE);
   }
   // Check for residual bits < CHUNK_SIZE
   k= n & CHUNK_MASK;
   if (k > 0)
   {
      cb.loadSh1(pRow[0]+i, pRow[1]+i, rowStride); // ensure LSB aligned?
      cb.add(bpfd, k); //dbg[2]+= k;
   }
} // addRowBPFD

__device__ void zeroBins (uint bpfd[], const int laneIdx, const int bins)
{
   for (int k= laneIdx; k < bins; k+= blockDim.x)
   {  // (transposed zeroing for write coalescing)
      for (int j= 0; j < blockDim.x; j++) { bpfd[j*bins+k]= 0; }
   }
} // zeroBins

#ifndef PACK16
__device__ void reduceBins (CUACount rBPFD[MKF_BINS], const uint bpfd[], const int laneIdx, const int bins)
{
   for (int k= laneIdx; k < bins; k+= blockDim.x)
   {  // (transposed reduction for read coalescing)
      CUACount t= 0;
      for (int j= 0; j < blockDim.x; j++) { t+= bpfd[j*bins+k]; }
      atomicAdd( rBPFD+k, t );
   }
} // reduceBins
#else
__device__ void reduceBins (CUACount rBPFD[MKF_BINS], const U16P bpfd[], const int laneIdx, const int bins)
{
   for (int k= laneIdx; k < bins; k+= blockDim.x)
   {  // (transposed reduction for read coalescing)
      CUACount t[2]= {0,0};
      for (int j= 0; j < blockDim.x; j++)
      {
         const U16P u= bpfd[j*bins+k];
         t[0]+= u & 0xFFFF;
         t[1]+= u >> 16;
      }
      const int i= k<<1;
      atomicAdd( rBPFD+i, t[0] );
      atomicAdd( rBPFD+i+1, t[1] );
   }
} // reduceBins
#endif

__global__ void addPlaneBPFD (CUACount rBPFD[MKF_BINS], const uint * pPln0, const uint * pPln1, const int rowStride, const int defW, const int defH)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x; // ???
   const int laneIdx= i & BPFD_BLKM;
   const int bins= BPFD_W32_BINS;
#ifndef PACK16
   __shared__ uint bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 32KB per warp
#else
   __shared__ U16P bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 16KB per warp
#endif
   //if (blockDim.x > BLKD) { printf("ERROR: addPlaneBPFD() - blockDim=%d", blockDim.x); return; }
   //else { printf(" - blockDim=%d,%d,%d\n", blockDim.x, blockDim.y, blockDim.z); }
   zeroBins(bpfd, laneIdx, bins);
   if (i < defH)
   {
      const U32 * pRow[2]= { pPln0 + i*rowStride, pPln1 + i*rowStride };

      addRowBPFD(bpfd+laneIdx*bins, pRow, rowStride, defW);
   }
   __syncthreads();
   reduceBins(rBPFD, bpfd, laneIdx, bins);
} // addPlaneBPFD


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

extern "C" int mkfCUDAGetBPFDSimple (Context *pC, const int def[3], const BinMapF32 *pBM)
{
   cudaError_t r;
   int blkD= BPFD_BLKD;//256;
   int nBlk= 0;

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
         if (pC->pDZ) { cudaMemset(pC->pDZ, 0, pC->bytesZ); }
         if (pC->nF <= blkD) { blkD= BPFD_BLKD; }
         nBlk= (pC->nF + blkD-1) / blkD;
         LOG("***\nmkfCUDAGetBPFDSimple() - bmc: %f,0x%X\n",pBM->t[0], pBM->m);
         // CAVEAT! Treated as 1D
         vThresh32<<<nBlk,blkD>>>(pC->pDU, pC->pDF, pC->nF, *pBM);
         ctuErr(NULL, "vThresh32()");
         cudaDeviceSynchronize();

         if (pC->pHU)
         {
            LOG("cudaMemcpy(%p, %p, %u)\n", pC->pHU, pC->pDU, pC->bytesU);
            r= cudaMemcpy(pC->pHU, pC->pDU, pC->bytesU, cudaMemcpyDeviceToHost);
            ctuErr(NULL, "{vThresh32+} cudaMemcpy()");
         }
      }
   }
   else if (pC->pHU)
   {
      r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice);
      if (NULL == pC->pDZ)
      {
         r= cudaMalloc(&(pC->pDZ), pC->bytesZ);
         ctuErr(&r, "cudaMalloc()");
      }
      if (pC->pDZ) { cudaMemset(pC->pDZ, 0, pC->bytesZ); }
   }
   if (pC->pDU && pC->pDZ)
   {
      //size_t bpdBytes= 256*sizeof(uint);
      //if ((pC->pDZ) && (pC->bytesZ >= bpdBytes))
      CUACount *pBPFD= (CUACount*)(pC->pDZ);
      const int rowStride= def[0] / 32;
      const int nRowPairs= def[1]-1;
      const int nPlanePairs= def[2]-1;
      const int planeStride= def[1] * rowStride;

      //if (nRowPairs <= blkD) {
      blkD= BPFD_BLKD;
      nBlk= (nRowPairs + blkD-1) / blkD;

      for (int i= 0; i < nPlanePairs; i++)
      {
         const U32 *pP0= pC->pDU + i * planeStride;
         const U32 *pP1= pC->pDU + (i+1) * planeStride;
         addPlaneBPFD<<<nBlk,blkD>>>(pBPFD, pP0, pP1, rowStride, def[0], nRowPairs); //, pBPFD+256);
         if (0 != ctuErr(NULL, "addPlane"))
         { LOG(" .. <<<%d,%d>>>(%p, %p, %p ..)\n", nBlk, blkD, pBPFD, pP0, pP1); }
      }
      cudaDeviceSynchronize();
      if (pC->pHZ)
      {
         r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost);
         ctuErr(&r, "{addPlane+} cudaMemcpy()");
      }
   }
   return(1);
} // mkfCUDAGetBPFDSimple


#ifdef MKF_CUDA_MAIN

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

#include "geomHacks.h"
#include "mkfUtil.h"

int buffAlloc (Context *pC, const int def[3], const int blkZ)
{
   int vol= def[0] * def[1] * def[2];

   pC->nF= vol;
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= BITS_TO_WRDSH(vol,5);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= BPFD_BLKD + blkZ * 256;
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

size_t bitCountNU32 (U32 u[], const int n)
{
   size_t t= 0;
   for (int i= 0; i<n; i++) { t+= BIT_COUNT_Z(u[i]); }
   return(t);
} // bitCountNU32

size_t mkft (Context *pC, const int def[3])
{
   BinMapF32 bmc;
   size_t sum= 0;

   //dumpF(pC->pHF+n, n, def[0]);
   setBinMapF32(&bmc,">=",0.5);
   mkfCUDAGetBPFDSimple(pC, def, &bmc);
#if 0
   size_t t= bitCountNU32(pC->pHU, pC->bytesU>>2);
   LOG("bitCountNU32() -> %zu\n", t);
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
      const size_t *pBPFD= (size_t*)pC->pHZ;
      LOG("\tvolFrac=%G chiEP=%G\n", volFrac(pBPFD), chiEP3(pBPFD));

      for (int i= 0; i<MKF_BINS; i++)
      {
         sum+= pBPFD[i];
         if (pBPFD[i] > 0) { LOG("[0x%X]=%u\n", i, pBPFD[i]); }
      }
   }
   return(sum);
} // mkft

int main (int argc, char *argv[])
{
   const int def[3]= {256,256,2};
   Context cux={0};

   if (buffAlloc(&cux, def, 1))
   {
      const size_t nC= prodSumA1VN(def,-1,3);
      LOG("[%d][%d][%d] -> %zu\n", def[0],def[1],def[2],nC);
      //sanityTest(&cux);
      if (0)
      {
         genPattern(cux.pHF, 1, def, 0.5*def[1] - 0.5);
         mkft(&cux,def);
      }
      else
      {
         Context t= cux;
         const int wDef= def[0] >> 5;
         const int lDef= def[1] * def[2];
         t.pHF= t.pDF= NULL;

         for (int i= 0; i < lDef; i++)
         {
            for (int j=0; j<wDef; j++) { t.pHU[wDef * i + j]= 0xFFFFFFFF; }
            t.pHU[wDef * i]= 0x7FFFFFFF; // NB: L to R order -> bits 0 to 31
         }
         mkft(&t,def);
         for (int i= 0; i < lDef; i++)
         {
            for (int j=0; j<wDef; j++) { t.pHU[wDef * i + j]= 0xFFFFFFFF; }
            t.pHU[wDef * i+1]= 0xFFFFFFFE;
         }
         mkft(&t,def);
      }
      cuBuffRelease(&cux);
   }
   cudaDeviceReset();
} // main

#endif // MKF_CUDA_MAIN
