// mkfCUDA.cu - Minkowski Functional pattern processing using CUDA NB: .cu assumes c++ style compilation
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Oct 2019

#include "mkfCUDA.h"
#include "utilCUDA.hpp"

/***/

// CUDA kernels and wrappers

#define PACK16

// Double warp if permitted by local mem & algorithm (16bit packed counters)
#ifdef PACK16
#define BPFD_W32_BINS (MKF_BINS/2)
#define BPFD_BLKS 6
#else
#define BPFD_W32_BINS MKF_BINS
//#define BPFD_BLKS 5
#endif

// Default single warp
#ifndef BPFD_BLKS
#define BPFD_BLKS 5
#endif
#define BPFD_BLKD (1<<BPFD_BLKS)
#define BPFD_BLKM (BPFD_BLKD-1)

// Wide counter for atomicAdd (nvcc dislikes size_t)
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
   __device__ ChunkBuf (const uint * __restrict__ pR0, const uint * __restrict__ pR1, const BMStride rowStride)
   {
      u00= pR0[0];
      u01= pR0[rowStride];
      u10= pR1[0];
      u11= pR1[rowStride];
      c= 0;
   } // ChunkBuf

   __device__ void loadSh1 (const uint * __restrict__ pR0, const uint * __restrict__ pR1, const BMStride rowStride)
   {
      //if (0 == pR1[rowStride]) { printf("%p+%x\n",pR1,rowStride); }
      u00|= ( (ULL) pR0[0] ) << 1;
      u01|= ( (ULL) pR0[rowStride] ) << 1;
      u10|= ( (ULL) pR1[0] ) << 1;
      u11|= ( (ULL) pR1[rowStride] ) << 1;
   } // loadSh1

#ifndef PACK16
   __device__ void add (uint bpfd[], const uint n)
   {
      for (uint i= 0; i < n; i++) { bpfd[ buildNext() ]++; }
   } // add (uint)
#else
   __device__ void add (U16P bpfd[], const uint n)
   {
      //const ushort2 lh[2]={ushort2(1,0),ushort2(0,1)}; //
      const U16P lh[2]={1,1<<16}; // even -> lo, odd -> hi (16b)

      for (uint i= 0; i < n; i++)
      {
         const uint bp= buildNext();
         //c++; if (bp < 0xFF) { logErr(c, u00,u01,u10,u11); } // DBG
         bpfd[ bp >> 1 ]+= lh[bp & 1];
      }
   } // add (U16P)
#endif
}; // class ChunkBuf

__device__ void addRowBPFD
(
   uint          bpfd[], // result pattern distribution
   const BMPackWord * pRow[2],
   const BMStride   rowStride,
   const uint       n    // Number of single bit elements packed in row
)
{  // seq
   const int m= n>>CHUNK_SHIFT;
   int i= 0, k= MIN(CHUNK_SIZE, n);
   ChunkBuf  cb(pRow[0]+0, pRow[1]+0, rowStride);

   cb.add(bpfd, k-1);
   // Subsequent whole chunks yield k patterns
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

__device__ void zeroBins (uint bpfd[], const uint laneIdx, const uint bins)
{
   for (uint k= laneIdx; k < bins; k+= blockDim.x)
   {  // (transposed zeroing for write coalescing)
      for (uint j= 0; j < blockDim.x; j++) { bpfd[j*bins+k]= 0; }
   }
} // zeroBins

#ifndef PACK16
__device__ void reduceBins (ULL rBPFD[MKF_BINS], const uint bpfd[], const uint laneIdx, const uint bins)
{
   for (uint k= laneIdx; k < bins; k+= blockDim.x)
   {  // (transposed reduction for read coalescing)
      ULL t= 0;
      for (int j= 0; j < blockDim.x; j++) { t+= bpfd[j*bins+k]; }
      atomicAdd( rBPFD+k, t );
   }
} // reduceBins
#else
__device__ void reduceBins (ULL rBPFD[MKF_BINS], const U16P bpfd[], const uint laneIdx, const uint bins)
{
   for (uint k= laneIdx; k < bins; k+= blockDim.x)
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

// Pairs of planes that may be non-contiguous i.e. wrap-around in partial buffer
// Consider:
//    using modulo operation on plane indices could be useful for more
//    efficient partial buffer batches. But how to set up ???
__global__ void addPlaneBPFD (ULL rBPFD[MKF_BINS], const BMPackWord * pWP0, const BMPackWord * pWP1, const BMOrg bmo)
{
   const uint i= blockIdx.x * blockDim.x + threadIdx.x; // ???
   const uint laneIdx= i & BPFD_BLKM;
   const uint bins= BPFD_W32_BINS;
#ifndef PACK16
   __shared__ uint bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 32KB/Warp -> 1W per SM
#else
   __shared__ U16P bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 16KB/Warp -> 2W per SM
#endif
   //if (blockDim.x > BLKD) { printf("ERROR: addPlaneBPFD() - blockDim=%d", blockDim.x); return; }
   //else { printf(" - blockDim=%d,%d,%d\n", blockDim.x, blockDim.y, blockDim.z); }
   zeroBins(bpfd, laneIdx, bins);
   if (i < bmo.rowPairs)
   {
      const BMPackWord * pRow[2]= {
         pWP0 + blockIdx.y * bmo.planeWS + i * bmo.rowWS,
         pWP1 + blockIdx.y * bmo.planeWS + i * bmo.rowWS
         };

      addRowBPFD(bpfd+laneIdx*bins, pRow, bmo.rowWS, bmo.rowElem);
   }
   __syncthreads();
   reduceBins(rBPFD, bpfd, laneIdx, bins);
} // addPlaneBPFD

// Contiguous sequence of planes i.e. entire full size buffer
__global__ void addMultiPlaneSeqBPFD (ULL rBPFD[MKF_BINS], const BMPackWord * pW, const BMOrg bmo)
{
   const uint i= blockIdx.x * blockDim.x + threadIdx.x; // ???
   //const uint j= blockIdx.y; // * blockDim.y + threadIdx.y;
   const uint laneIdx= i & BPFD_BLKM;
   const uint bins= BPFD_W32_BINS;
#ifndef PACK16
   __shared__ uint bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 32KB/Warp -> 1W per SM
#else
   __shared__ U16P bpfd[BPFD_W32_BINS*BPFD_BLKD]; // 16KB/Warp -> 2W per SM
#endif
   zeroBins(bpfd, laneIdx, bins);
   if (i < bmo.rowPairs)
   {
      const BMPackWord * pRow[2];
      pRow[0]= pW + blockIdx.y * bmo.planeWS + i * bmo.rowWS;
      pRow[1]= pRow[0] + bmo.planeWS;
      addRowBPFD(bpfd+laneIdx*bins, pRow, bmo.rowWS, bmo.rowElem);
   }
   __syncthreads();
   reduceBins(rBPFD, bpfd, laneIdx, bins);
} // addMultiPlaneSeqBPFD

/***/

extern "C"
size_t * mkfCUDAGetBPFD (size_t * pBPFD, const BMOrg *pO, const BMPackWord * pW, const int profile)
{
   const int blkD= BPFD_BLKD;
   const int nBlk= (pO->rowPairs + blkD-1) / blkD;

   CTimerCUDA t;
   //t.stampStream();
   //LOG("\tsd= %u, %u\n", pO->rowWS, pO->planeWS);
   if (NULL == pBPFD) { return(NULL); }

   switch (profile)
   {
      case MKFCU_PROFILE_FAST :
      {
         dim3 grd(nBlk,pO->planePairs,1);
         dim3 blk(blkD,1,1);
         addMultiPlaneSeqBPFD<<<grd,blk>>>((ULL*)pBPFD, pW, *pO);
         if (0 != ctuErr(NULL, "addMultiPlaneSeqBPFD"))
         { LOG(" .. <<<(%d,%d)(%d)>>>(%p, %p ..)\n", grd.x, grd.y, blk.x, pBPFD, pW); }
         break;
      }

      case MKFCU_PROFILE_FLEX :
      {
         dim3 grd(nBlk,pO->planePairs,1);
         dim3 blk(blkD,1,1);
         addPlaneBPFD<<<grd,blk>>>((ULL*)pBPFD, pW, pW + pO->planeWS, *pO); //, pBPFD+256);
         if (0 != ctuErr(NULL, "addPlaneBPFD"))
         { LOG(" .. <<<(%d,%d)(%d)>>>(%p, %p, %p ..)\n", grd.x, grd.y, blk.x, pBPFD, pW, pW + pO->planeWS); }
         break;
      }

      default :
      {
         for (int i= 0; i < pO->planePairs; i++)
         {
            const BMPackWord *pP0= pW + i * pO->planeWS;
            const BMPackWord *pP1= pW + (i+1) * pO->planeWS;
            //LOG(" RP: %d %d*%d=0x%08X, %p, %p \n", rowStride, planeStride, sizeof(*pP0), planeStride*sizeof(*pP0), pP0, pP1);
            addPlaneBPFD<<<nBlk,blkD>>>((ULL*)pBPFD, pP0, pP1, *pO); //, pBPFD+256);
            if (0 != ctuErr(NULL, "addPlaneBPFD"))
            { LOG(" .. <<<%d,%d>>>(%p, %p, %p ..)\n", nBlk, blkD, pBPFD, pP0, pP1); }
         }
/*         const BMPackWord *pP0= pW;
         const BMPackWord *pP1= pP0 + pO->planeWS;

         for (int i= 0; i < pO->planePairs; i++)
         {
            addPlaneBPFD<<<nBlk,blkD>>>((ULL*)pBPFD, pP0, pP1, *pO);
            { LOG(" .. <<<%d,%d>>>(%p, %p, %p ..)\n", nBlk, blkD, pBPFD, pP0, pP1); }
            pP0= pP1; pP1+= pO->planeWS;
         }*/
         break;
      }
   }
   LOG("mkfCUDAGetBPFD() - dt= %Gms\n", t.elapsedms());
   //cudaDeviceSynchronize();
   return(pBPFD);
} // mkfCUDAGetBPFD

extern "C"
int mkfCUDAGetBPFDautoCtx (Context *pC, const int def[3], const BinMapF64 *pM, const int profHack)
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
         ConstFieldPtr fieldPtr[BMCU_FIELD_MAX];
         BMFieldInfo fi;
         setupFields(&fi, autoFieldPtr(fieldPtr, pC), pC->nField, def, pC->bytesElem, profHack);
         if (NULL == binMapCUDA(pC->pDU, &(pC->bmo), &fi, pM))
         {
            LOG("\tpC= %p; pC->pDF= %p; &(pC->pDF)= %p\n\tpFDPT= %p; pFDPT->p= %p\n", pC, pC->pDF, &(pC->pDF),
               fi.pFieldDevPtrTable, fi.pFieldDevPtrTable->p);
            return(0);
         }
         else if (pC->pHU)
         {
            LOG("cudaMemcpy(%p, %p, %u)\n", pC->pHU, pC->pDU, pC->bytesU);
            r= cudaMemcpy(pC->pHU, pC->pDU, pC->bytesU, cudaMemcpyDeviceToHost);
            ctuErr(NULL, "{binMapCUDA+} cudaMemcpy()");
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
   }
   if (pC->pDU && pC->pDZ)
   {
      cudaMemset(pC->pDZ, 0, pC->bytesZ); // Kernel will add to BPFD so start clean
      mkfCUDAGetBPFD((size_t*)(pC->pDZ), &(pC->bmo), pC->pDU, 2);
      if (pC->pHZ)
      {
         LOG("cudaMemcpy(%p, %p, %u)\n", pC->pHZ, pC->pDZ, pC->bytesZ);
         r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost);
         ctuErr(&r, "{mkfCUDAGetBPFD+} cudaMemcpy()");
      }
   }
   return(1);
} // mkfCUDAGetBPFDautoCtx

// Internal test code
#ifdef MKF_CUDA_MAIN

#include "geomHacks.h"
#include "mkfUtil.h"

int buffAlloc (Context *pC, const int def[3], const NumEnc e, int flags)
{
   int b;

   pC->enc= e;
   pC->nElem= prodNI(def,3);
   pC->nField= 1; //MAX(1,n);
   pC->bytesF= encSizeN(&b, pC->nElem * pC->nField, e);
   if ((flags & 0x01) && (b > 0))
   {
      pC->bytesElem= BITS_TO_BYTES(b);
   }
   else { pC->nElem= 0; pC->nField= 0; pC->bytesElem= 0; pC->bytesF= 0; }

   if (flags & 0x02)
   {
      pC->nU= setBMO(&(pC->bmo), def, 0);
      pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   }
   else { pC->nU= 0; pC->bytesU= 0; }

   pC->nZ= MKF_BINS;
   pC->bytesZ= sizeof(size_t) * pC->nZ;

   LOG("F: %d * %d -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nField, pC->nElem, pC->bytesF, pC->nU, pC->bytesU);

   return cuBuffAlloc(pC,0);
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
size_t mkftu (const Context *pC, const int def[3], const float mScale, const uint8_t profHack)
{
   cudaError_t r;
   size_t sum= 0;
   int verbose= 0;

   r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice); ctuErr(&r, "cudaMemcpy(pDU, pHU)");
   r= cudaMemset(pC->pDZ, 0, pC->bytesZ); ctuErr(&r, "cudaMemset(pDZ)");
   mkfCUDAGetBPFD((size_t*)(pC->pDZ), &(pC->bmo), pC->pDU, profHack);
   r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost); ctuErr(&r, "cudaMemcpy(pHZ, pDZ)");

   const size_t *pBPFD= (size_t*)pC->pHZ;
   float m[4];
   for (int i= 0; i<MKF_BINS; i++)
   {
      sum+= pBPFD[i];
      if (verbose && (pBPFD[i] > 0)) { LOG("[0x%X]=%u\n", i, pBPFD[i]); }
   }
   if (mkfMeasureBPFD(m, pBPFD, mScale, 0))
   {
      LOG("\tID%d: S=%zu\tK M S V: %G %G %G %G\n", profHack, sum, m[0],m[1],m[2],m[3]);
   }
   return(sum);
} // mkftu

#define PAT_ID 3
int main (int argc, char *argv[])
{
   const int def[3]= {256, 256, 256}; // {64,64,2}; //{96,9,9};
   const float param[]= { def[0] * 0.75, 1, 0 }; //256-64;
   Context cux={0};
   const float mScale= 3.0 / sumNI(def,3); // reciprocal mean
   size_t n;

   if (buffAlloc(&cux, def, ENC_F32, 0x03))
   {
      const size_t nC= prodOffsetNI(def,3,-1);
      LOG("[%d][%d][%d] -> %zuCells\n", def[0],def[1],def[2],nC);
      //sanityTest(&cux);
      if (cux.pHF)
      {
         BinMapF32 mc={0};//ENC_F32, 1,
         n= genPattern(NULL, cux.pHF, def, cux.enc, cux.nField, PAT_ID, param);
         LOG("genPattern(.. ENC_F32 ..) - n=%zu PCVF=%G\n", n, (float)n / cux.nElem);
         mkfCUDAGetBPFDautoCtx(&cux, def, setBinMapF32(&mc,">=",0.5), 0x00);
         const size_t *pBPFD= (size_t*)(cux.pHZ);
         float m[4];
         if (mkfMeasureBPFD(m, pBPFD, mScale, 0))
         {
            LOG("\t[F]:\tK M S V: %G %G %G %G\n", m[0],m[1],m[2],m[3]);
            LOG("\tbitCountNU32()=%zu\n", bitCountNU32(cux.pHU, cux.nU));
         }
      }
      if (cux.pHU)
      {
         memset(cux.pHU, 0, cux.bytesU);
#if 0    // word boundary test - should be factored into pattern gen
         const int wDef= cux.sd.row;
         const int lDef= def[1] * def[2];
         const uint m= BIT_MASK(def[0] & 0x1F);
         LOG("mkfCUDA - main() - [%d,%d,%d] -> [%d,%d]\n", def[0], def[1], def[2], wDef, lDef);
         for (int i= 0; i < lDef; i++)
         {  // NB: L to R order -> bits 0 to 31
            for (int j=0; j<wDef; j++) { cux.pHU[wDef * i + j]= 0xFFFFFFFF; }
            if (m) { cux.pHU[wDef * (i+1) - 1]= m; }
         }
         mkftu(&cux,def, mScale);

         int j= wDef/2;
         for (int i= 0; i < lDef; i++)
         {
            cux.pHU[wDef * i + j]= 0xFFF7FFFF;
         }
         for (int i= 0; i < lDef; i++)
         {
            int j= 0;
            for ( ; j<wDef-1; j++) { LOG("%08X ", cux.pHU[wDef * i + j]); }
            LOG("%08X\n", cux.pHU[wDef * i + j]);
         }
#endif
         if (0 != (def[0] & 0x1F)) { WARN("MKF_CUDA_MAIN / genPattern() - unpadded def[0]=%d\n", def[0]); }
         n= genPattern(NULL, cux.pHU, def, ENC_U1, 1, PAT_ID, param);
         LOG("genPattern(.. ENC_U1 ..) - n=%zu\n", n);
         mkftu(&cux,def,mScale,2);
         mkftu(&cux,def,mScale,3);
      }
      cuBuffRelease(&cux);
   }
   cudaDeviceReset();
} // main

#endif // MKF_CUDA_MAIN
