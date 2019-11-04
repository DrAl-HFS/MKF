// bpfdCUDA.cuh - CUDA device classes/functions for Packed Binary Map reduction to Binary Pattern Frequency Distribution
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Nov 2019

#ifndef BPFD_CUDA_CUH
#define BPFD_CUDA_CUH

//define PACK16
#define WARP_SHIFT   (5)
//if ((1<<WARP_SHIFT) != warpSize) ERROR!

// Double warp if permitted by local mem & algorithm
#ifdef PACK16
// Fully privatised (per thread) using 16bit packed counters (double warp -> 32KB sh.mem.)
#define BPFD_W32_BINS (MKF_BINS/2)
#define BPFD_BLKS 6
#define BPFD_NSHD (1<<BPFD_BLKS)
#else
// Unpacked with warp-level privatisation (eight warps -> 32KB sh.mem.)
#define BPFD_W32_BINS MKF_BINS
#define BPFD_BLKS 8
#define BPFD_NSHD (1<<(BPFD_BLKS-5)) // =(BPFD_BLKD/warpSize)
#endif

#ifndef BPFD_BLKS // Default is single warp privatised
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


class ChunkBuf
{
   ULL u00, u01, u10, u11; // 64bit not ideal as arch. is 32bit, but 8 or 16bit chunks need revised BinMap (complicated?)

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
   } // ChunkBuf

   __device__ void loadSh1 (const uint * __restrict__ pR0, const uint * __restrict__ pR1, const BMStride rowStride)
   {
      u00|= ( (ULL) pR0[0] ) << 1;
      u01|= ( (ULL) pR0[rowStride] ) << 1;
      u10|= ( (ULL) pR1[0] ) << 1;
      u11|= ( (ULL) pR1[rowStride] ) << 1;
   } // loadSh1

#ifdef PACK16
   __device__ void add (U16P bpfd[], const uint n)
   {  // Fully privatised result
      //const ushort2 lh[2]={ushort2(1,0),ushort2(0,1)}; //
      const U16P lh[2]={1,1<<16}; // even -> lo, odd -> hi (16b)

      for (uint i= 0; i < n; i++)
      {
         const uint bp= buildNext();
         //c++; if (bp < 0xFF) { logErr(c, u00,u01,u10,u11); } // DBG
         bpfd[ bp >> 1 ]+= lh[bp & 1];
      }
   } // add (U16P)
#else
   __device__ void add (uint bpfd[], const uint n)
   {  // Shared result
      for (uint i= 0; i < n; i++) { atomicAdd( bpfd + buildNext(), 1); } // { bpfd[ buildNext() ]++; }
   } // add (uint)
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

__device__ void zeroW (uint w[], const uint laneIdx, const uint nW)
{
   for (uint k= laneIdx; k < nW; k+= blockDim.x) { w[k]= 0; }
} // zeroW

#ifdef PACK16
__device__ void reduceBins (ULL rBPFD[MKF_BINS], const U16P bpfd[], const uint laneIdx, const uint nD)
{
   for (uint k= laneIdx; k < BPFD_W32_BINS; k+= blockDim.x)
   {  // (transposed reduction for read coalescing)
      ULL t[2]= {0,0};
      for (int j= 0; j < nD; j++)
      {
         const U16P u= bpfd[j*BPFD_W32_BINS+k];
         t[0]+= u & 0xFFFF;
         t[1]+= u >> 16;
      }
      const int i= k<<1;
      atomicAdd( rBPFD+i, t[0] );
      atomicAdd( rBPFD+i+1, t[1] );
   }
} // reduceBins
#else
__device__ void reduceBins (ULL rBPFD[MKF_BINS], const uint bpfd[], const uint laneIdx, const uint nD)
{
   for (uint k= laneIdx; k < BPFD_W32_BINS; k+= blockDim.x)
   {  // (transposed reduction for read coalescing)
      ULL t= 0;
      for (int j= 0; j < nD; j++) { t+= bpfd[j*BPFD_W32_BINS+k]; }
      atomicAdd( rBPFD+k, t );
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
#ifdef PACK16
   __shared__ U16P bpfd[BPFD_W32_BINS*BPFD_NSHD]; // 16KB/Warp -> 2W per SM
   const uint distIdx= laneIdx*BPFD_W32_BINS;
#else
   __shared__ uint bpfd[BPFD_W32_BINS*BPFD_NSHD]; // 32KB/Warp -> 1W per SM
   const uint distIdx= (laneIdx >> WARP_SHIFT) * BPFD_W32_BINS;
#endif
   zeroW(bpfd, laneIdx, BPFD_W32_BINS*BPFD_NSHD);
   __syncthreads();
   if (i < bmo.rowPairs)
   {
      const BMPackWord * pRow[2];
      pRow[0]= pWP0 + blockIdx.y * bmo.planeWS + i * bmo.rowWS;
      pRow[1]= pWP1 + blockIdx.y * bmo.planeWS + i * bmo.rowWS;
      addRowBPFD(bpfd+distIdx, pRow, bmo.rowWS, bmo.rowElem);
   }
   __syncthreads();
   reduceBins(rBPFD, bpfd, laneIdx, BPFD_NSHD);
} // addPlaneBPFD

// Contiguous sequence of planes i.e. entire full size buffer
__global__ void addMultiPlaneSeqBPFD (ULL rBPFD[MKF_BINS], const BMPackWord * pW, const BMOrg bmo)
{
   const uint i= blockIdx.x * blockDim.x + threadIdx.x; // ???
   //const uint j= blockIdx.y; // * blockDim.y + threadIdx.y;
   const uint laneIdx= i & BPFD_BLKM;
#ifdef PACK16
   __shared__ U16P bpfd[BPFD_W32_BINS*BPFD_NSHD]; // 16KB/Warp -> 2W per SM
   const uint distIdx= laneIdx*BPFD_W32_BINS;
#else
   __shared__ uint bpfd[BPFD_W32_BINS*BPFD_NSHD]; // 32KB/Warp -> 1W per SM
   const uint distIdx= (laneIdx >> WARP_SHIFT) * BPFD_W32_BINS;
#endif
   zeroW(bpfd, laneIdx, BPFD_W32_BINS*BPFD_NSHD);
   __syncthreads();
   if (i < bmo.rowPairs)
   {
      const BMPackWord * pRow[2];
      pRow[0]= pW + blockIdx.y * bmo.planeWS + i * bmo.rowWS;
      pRow[1]= pRow[0] + bmo.planeWS;
      addRowBPFD(bpfd+distIdx, pRow, bmo.rowWS, bmo.rowElem);
   }
   __syncthreads();
   reduceBins(rBPFD, bpfd, laneIdx, BPFD_NSHD);
} // addMultiPlaneSeqBPFD

#endif // BPFD_CUDA_CUH
