// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Sept 2019

#include "binMapCUDA.h"
#include "utilCUDA.hpp"


#define VT_BLKS 5
#define VT_BLKN (1<<VT_BLKS)
#define VT_BLKM (VT_BLKN-1)

/***/

/*
__global__ void vThresh8 (uint r[], const float f[], const size_t n, const BinMapF32 mc)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[VT_BLKN];
   if (i < n)
   {
      const int j= i & VT_BLKM;
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
            r[i>>VT_BLKS]= ( z[0] << 0 ) | ( z[8] << 8 ) | ( z[16] << 16 ) | ( z[24] << 24 );
         }
      }
   }
} // vThresh8
*/
// class ??
__device__ int bm1f32 (const float f, const BinMapF32& bm)
{
   const int d= (1 + (f > bm.t[0]) - (f < bm.t[0]));
   return( (bm.m >> d) & 0x1 );
} // bm1f32

__device__ void merge32 (uint u[32], const int j)
{
/* TODO: consider using CUDA9 warp level primitives...
#define FULL_MASK 0xffffffff
for (int offset = 16; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
*/
   //__syncthreads(); // Unnecessary - no divergence at this point

   if (0 == (j & 0x3))
   {  // j : { 0, 4, 8, 12, 16, 10, 24, 28 } 8P 3I
      for (int l=1; l<4; l++) { u[j]|= u[j+l]; }

      __syncthreads(); // Required for (unexplained) divergence

      if (0 == (j & 0xF))
      {  // j : { 0, 16 } 2P 3I
         for (int l=4; l<16; l+=4) { u[j]|= u[j+l]; }

         __syncthreads(); //  Optional ?
      }
   }
} // merge32

/***/

__global__ void vThreshL32 (uint r[], const float f[], const size_t n, const BinMapF32 bm)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[VT_BLKN];
   if (i < n)
   {
      const int j= i & VT_BLKM;

      z[j]= bm1f32(f[i],bm) << j;

      merge32(z, j);
      if (0 == j) { r[i>>VT_BLKS]= z[0] | z[16]; }
/*
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
*/
   }
} // vThreshL32

__global__ void vThreshP32 (uint r[], const float f[], const size_t nX, const size_t nY, const BinMapF32 bm)
{
   const size_t x= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint u[VT_BLKN];
   if (x < nX)// && (y < nY))
   {
      //const size_t y= blockIdx.y * blockDim.y + threadIdx.y;
      const size_t i= x + nX * threadIdx.y;
      const int j= i & VT_BLKM;

      u[j]= bm1f32(f[i],bm) << j;

      merge32(u, j);
      if (0 == j) { r[i>>VT_BLKS]= u[0] | u[16]; }
   }
} // vThreshP32

__global__ void vThreshSum (uint r[], const MultiFieldDesc mfd, const int nS, const size_t n, const BinMapF32 bm)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
//   const size_t j= blockIdx.y * blockDim.y + threadIdx.y;
//   const size_t k= blockIdx.z * blockDim.z + threadIdx.z;
   __shared__ uint u[VT_BLKN];
   if (i < n)
   {
      const int j= i & VT_BLKM;
      float s= (mfd.pF32[0])[i];
      for (int k=1; k<nS; k++) { s+= (mfd.pF32[k])[i]; }
      u[j]= bm1f32(s,bm) << j;

      merge32(u, j);
      if (0 == j) { r[i>>VT_BLKS]= u[0] | u[16]; }
   }
} // vThreshSum


/* INTERFACE */

extern "C"
void binMapCudaRowsF32
(
   U32 * pBM,
   const F32 * pF,
   const int rowLenF,      // row length ie. "X dimension"
   const int rowStrideBM,  // 32bit word stride of rows of packed binary map, should be >= rowLenF/32
   const int nRows,        // product of other "dimensions" (Y * Z)
   const BinMapF32 *pMC
)
{
   CTimerCUDA t;
   //t.stampStream();

   if (0 == (rowLenF & VT_BLKM))
   {  // Treat as 1D
      size_t nF= rowLenF * nRows;
      vThreshL32<<<nF/VT_BLKN,VT_BLKN>>>(pBM, pF, nF, *pMC);
   }
   else
   {  // Hacky 2D - needs proper implementation
      int nBlkRow= (rowLenF + VT_BLKM) / VT_BLKN;
#if 1
      int rowStride= BITS_TO_WRDSH(rowLenF,5); //(rowLenF + 31) / 32;
      for (int i=0; i<nRows; i++)
      {
         vThreshL32<<<nBlkRow,VT_BLKN>>>(pBM + i * rowStride, pF + i * rowLenF, rowLenF, *pMC);
      }
#else
      vThreshP32<<<dim3(nBlkRow,nRows),dim3(VT_BLKN)>>>(pBM, pF, rowLenF, nRows, *pMC);
#endif
   }
   LOG("binMapCudaRowsF32(.., rowLen=%d, BM(%f,0x%X) ) - dt= %Gms\n", rowLenF, pMC->t[0], pMC->m, t.elapsedms());
   //ctuErr(NULL, "vThresh32()");
   //cudaDeviceSynchronize(); // stream sync provided by timer
} // binMapCudaRowsF32

extern "C"
BMStrideDesc *binMapCUDA
(
   uint        * pBM,
   BMStrideDesc * pBMSD,
   const MultiFieldInfo * pMFI,
   const BinMapF32      * pMC
)
{
   if (0 == (pMFI->def[0] & 0x1F))
   {
      //vThreshSum<<<nBlk,VT_BLKN>>>(pBM, pMFI->mfd, pMFI->nField, nF, *pMC);

      //return(pBMSD);
   }
   return(NULL);
} // binMapCUDA
