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


#define PACK16

#ifdef PACK16
#define BPFD_W32_BINS (MKF_BINS/2)
#define BPFD_BLKS 6
#else
#define BPFD_W32_BINS MKF_BINS
//#define BPFD_BLKS 5
#endif

#ifndef BPFD_BLKS
#define BPFD_BLKS 5
#endif
#define BPFD_BLKD (1<<BPFD_BLKS)
#define BPFD_BLKM (BPFD_BLKD-1)

typedef unsigned long long ULL;
typedef uint   U16P; // ushort2 ???

#define CHUNK_SHIFT (5)
#define CHUNK_SIZE (1<<CHUNK_SHIFT)
#define CHUNK_MASK (CHUNK_SIZE-1)

//__device__ static int gBrkCount=0;
//__device__ void brk (void) { gBrkCount++; }
__device__ void logErr (ULL c, ULL u00, ULL u01, ULL u10, ULL u11)
{  // single printf unreliable, not sure this is either...
   printf("%u:", c);
   printf(" %016X",u00); printf(" %016X", u01);
   printf(" %016X", u10); printf(" %016X\n", u11);
} // logErr

class ChunkBuf
{
   ULL u00, u01, u10, u11, c;

   //friend logErr (ChunkBuf&);
   __device__ uint buildNext (void)
   {
      uint bp=  ( u00 & 0x3); u00 >>= 1;
      bp|= ((u01 & 0x3) << 2); u01 >>= 1;
      bp|= ((u10 & 0x3) << 4); u10 >>= 1;
      bp|= ((u11 & 0x3) << 6); u11 >>= 1;
      return(bp);
   } // buildNext

public:
   __device__ ChunkBuf (const uint * __restrict__ pR0, const uint * __restrict__ pR1, const int rowStride)
   {
      u00= pR0[0];
      u01= pR0[rowStride];
      u10= pR1[0];
      u11= pR1[rowStride];
      c= 0;
   } // ChunkBuf

   __device__ void loadSh1 (const uint * __restrict__ pR0, const uint * __restrict__ pR1, const int rowStride)
   {
      if (0 == pR1[rowStride]) { printf("%p+%x\n",pR1,rowStride); }
      u00|= ( (ULL) pR0[0] ) << 1;
      u01|= ( (ULL) pR0[rowStride] ) << 1;
      u10|= ( (ULL) pR1[0] ) << 1;
      u11|= ( (ULL) pR1[rowStride] ) << 1;
   } // loadSh1

#ifndef PACK16
   __device__ void add (uint bpfd[], const int n)
   {
      for (int i= 0; i < n; i++) { bpfd[ buildNext() ]++; }
   } // add (uint)
#else
   __device__ void add (U16P bpfd[], const int n)
   {
      //const ushort2 lh[2]={ushort2(1,0),ushort2(0,1)}; //
      const U16P lh[2]={1,1<<16}; // even -> lo, odd -> hi (16b)

      for (int i= 0; i < n; i++)
      {
         const uint bp= buildNext();
         c++; if (bp < 0xFF) { logErr(c, u00,u01,u10,u11); } // DBG
         bpfd[ bp >> 1 ]+= lh[bp & 1];
      }
   } // add (U16P)
#endif
}; // class ChunkBuf

__device__ void addRowBPFD
(
   uint          bpfd[], // result pattern distribution
   const uint * pRow[2],
   const int    rowStride,
   const int    n    // Number of single bit elements packed in row
)
{  // seq
   const int m= n>>CHUNK_SHIFT;
   int i= 0, k= MIN(CHUNK_SIZE, n);
   ChunkBuf  cb(pRow[0]+0, pRow[1]+0, rowStride);

   cb.add(bpfd, k-1);
   // Subsequent whole chunks yield n patterns
   while (++i < m)
   {
      cb.loadSh1(pRow[0]+i, pRow[1]+i, rowStride);
      cb.add(bpfd, k);
   }
   // Check for residual bits < CHUNK_SIZE
   k= n & CHUNK_MASK;
   if (k > 0)
   {
      cb.loadSh1(pRow[0]+m, pRow[1]+m, rowStride); // ensure LSB aligned?
      cb.add(bpfd, k);
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
__device__ void reduceBins (ULL rBPFD[MKF_BINS], const uint bpfd[], const int laneIdx, const int bins)
{
   for (int k= laneIdx; k < bins; k+= blockDim.x)
   {  // (transposed reduction for read coalescing)
      ULL t= 0;
      for (int j= 0; j < blockDim.x; j++) { t+= bpfd[j*bins+k]; }
      atomicAdd( rBPFD+k, t );
   }
} // reduceBins
#else
__device__ void reduceBins (ULL rBPFD[MKF_BINS], const U16P bpfd[], const int laneIdx, const int bins)
{
   for (int k= laneIdx; k < bins; k+= blockDim.x)
   {  // (transposed reduction for read coalescing)
      ULL t[2]= {0,0};
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

__global__ void addPlaneBPFD (ULL rBPFD[MKF_BINS], const uint * pPln0, const uint * pPln1, const int rowStride, const int defW, const int defH)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x; // ???
   const int laneIdx= i & BPFD_BLKM;
   const int bins= BPFD_W32_BINS;
#ifndef PACK16
   __shared__ uint bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 32KB/Warp -> 1W per SM
#else
   __shared__ U16P bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 16KB/Warp -> 2W per SM
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

extern "C"
int mkfCUDAGetBPFD (size_t * pBPFD, const int def[3], const U32 * pBM)
{
   const int rowStride= BITS_TO_WRDSH(def[0],5);
   const int nRowPairs= def[1]-1;
   const int nPlanePairs= def[2]-1;
   const int planeStride= def[1] * rowStride;

   const U32 *pP0= pBM;// + i * planeStride;
   const U32 *pP1= pP0 + planeStride; //pBM + (i+1) * planeStride;

   //if (nRowPairs <= blkD) {
   int blkD= BPFD_BLKD;
   int nBlk= (nRowPairs + blkD-1) / blkD;

   for (int i= 0; i < nPlanePairs; i++)
   {
      LOG(" RP: %d %d*%d=0x%08X, %p, %p \n", rowStride, planeStride, sizeof(*pP0), planeStride*sizeof(*pP0), pP0, pP1);
      addPlaneBPFD<<<nBlk,blkD>>>((ULL*)pBPFD, pP0, pP1, rowStride, def[0], nRowPairs); //, pBPFD+256);
      if (0 != ctuErr(NULL, "addPlaneBPFD"))
      { LOG(" .. <<<%d,%d>>>(%p, %p, %p ..)\n", nBlk, blkD, pBPFD, pP0, pP1); }
   }
   cudaDeviceSynchronize();
   return(MKF_BINS);
} // mkfCUDAGetBPFD

extern "C"
int mkfCUDAGetBPFDautoCtx (Context *pC, const int def[3], const BinMapF32 *pMC)
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
         if (pC->pDZ) { cudaMemset(pC->pDZ, 0, pC->bytesZ); }
         binMapCudaRowsF32(pC->pDU, pC->pDF, def[0], BITS_TO_WRDSH(def[0],5), def[1] * def[2], pMC);

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
      mkfCUDAGetBPFD((size_t*)(pC->pDZ), def, pC->pDU);
      if (pC->pHZ)
      {
         r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost);
         ctuErr(&r, "{addPlane+} cudaMemcpy()");
      }
   }
   return(1);
} // mkfCUDAGetBPFDautoCtx


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

#if 0
checkHU ()
{
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
}
#endif
size_t mkft (Context *pC, const int def[3])
{
   cudaError_t r;
   size_t sum= 0;

   r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice); ctuErr(&r, "cudaMemcpy(pDU, pHU)");
   r= cudaMemset(pC->pDZ, 0, pC->bytesZ); ctuErr(&r, "cudaMemset(pDZ)");
   mkfCUDAGetBPFD((size_t*)(pC->pDZ), def, pC->pDU);
   r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost); ctuErr(&r, "cudaMemcpy(pHZ, pDZ)");

   const size_t *pBPFD= (size_t*)pC->pHZ;
   LOG("\tvolFrac=%G chiEP=%G\n", volFrac(pBPFD), chiEP3(pBPFD));

   for (int i= 0; i<MKF_BINS; i++)
   {
      sum+= pBPFD[i];
      if (pBPFD[i] > 0) { LOG("[0x%X]=%u\n", i, pBPFD[i]); }
   }
   return(sum);
} // mkft

int main (int argc, char *argv[])
{
   const int def[3]= {257,3,3};
   Context cux={0};

   if (buffAlloc(&cux, def, 1))
   {
      const size_t nC= prodOffsetNI(def,3,-1);
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
         const int wDef= BITS_TO_WRDSH(def[0],5);
         const int lDef= def[1] * def[2];
         t.pHF= t.pDF= NULL;
         LOG("mkfCUDA - main() - [%d,%d,%d] -> [%d,%d]\n", def[0], def[1], def[2], wDef, lDef);
         for (int i= 0; i < lDef; i++)
         {  // NB: L to R order -> bits 0 to 31
            for (int j=0; j<wDef; j++) { t.pHU[wDef * i + j]= 0xFFFFFFFF; }
            t.pHU[wDef * (i+1) - 1]= 0x1; //0x80000000; //0x7FFFFFFF;
         }
         for (int i= 0; i < lDef; i++)
         {
            int j= 0;
            for ( ; j<wDef-1; j++) { LOG("%08X ", t.pHU[wDef * i + j]); }
            LOG("%08X\n", t.pHU[wDef * i + j]);
         }
         mkft(&t,def);
#if 0
         for (int i= 0; i < lDef; i++)
         {
            for (int j=0; j<wDef; j++) { t.pHU[wDef * i + j]= 0xFFFFFFFF; }
            t.pHU[wDef * i+1]= 0xFFFFFFFE;
         }
         mkft(&t,def);
#endif
      }
      cuBuffRelease(&cux);
   }
   cudaDeviceReset();
} // main

#endif // MKF_CUDA_MAIN
