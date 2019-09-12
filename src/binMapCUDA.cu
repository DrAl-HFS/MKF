// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Sept 2019

#include "binMapCUDA.h"
#include "utilCUDA.hpp"


#define VT_WRDS 5
#define VT_WRDN (1<<VT_WRDS)
#define VT_WRDM (VT_WRDN-1)

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
      if (0 == j) { r[i>>VT_WRDS]= z[0] | z[16]; }
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

__global__ void vThreshV32 (uint rBM[], const float f[], const int defX, const BMStrideDesc sBM, const Stride3 sF, const BinMapF32 bm)
{
   const int x= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint u[VT_BLKN];
   if (x < defX)
   {
      size_t i= x * sF.s[0] + blockIdx.y * sF.s[1] + blockIdx.z * sF.s[2];
      const int j= i & VT_BLKM;

      u[j]= bm1f32(f[i],bm) << j; // (j & VT_WRDM)

      merge32(u, j);
      if (0 == j) // & VT_WRDM) if BLKS > WRDS !
      {  // (x >> VT_WRDS)
         i= blockIdx.x + blockIdx.y * sBM.row + blockIdx.z * sBM.plane;
         rBM[i]= u[0] | u[16];
      }
   }
} // vThreshV32

__global__ void vThreshVSum32
(
   uint rBM[],
   const MultiFieldDesc mfd,
   const int nS,
   const int defX,
   const BMStrideDesc sBM,
   const Stride3 sF,
   const BinMapF32 bm
)
{
   const int x= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint u[VT_BLKN];
   if (x < defX)// && (y < defY))
   {
      size_t i= x * sF.s[0] + blockIdx.y * sF.s[1] + blockIdx.z * sF.s[2];
      const int j= i & VT_BLKM;

      float s= (mfd.pF32[0])[i];
      for (int k=1; k<nS; k++) { s+= (mfd.pF32[k])[i]; }

      u[j]= bm1f32(s,bm) << j;

      merge32(u, j);
      if (0 == j)
      {
         i= (x >> VT_WRDS) + blockIdx.y * sBM.row + blockIdx.z * sBM.plane;
         rBM[i]= u[0] | u[16];
      }
   }
} // vThreshVSum32


/* INTERFACE */

extern "C"
int binMapCudaRowsF32
(
   U32 * pBM,
   const F32 * pF,
   const int rowLenF,      // row length ie. "X dimension"
   const int rowStrideBM,  // 32bit word stride of rows of packed binary map, should be >= rowLenF/32
   const int nRows,        // product of other "dimensions" (Y * Z)
   const BinMapF32 *pMC
)
{
   int r= 0;
   CTimerCUDA t;

   if (0 == (rowLenF & VT_BLKM))
   {  // Treat as 1D
      size_t nF= rowLenF * nRows;
      vThreshL32<<<nF/VT_BLKN,VT_BLKN>>>(pBM, pF, nF, *pMC);
      r= (0 == ctuErr(NULL, "vThreshL32()"));
   }
   else
   {  // Hacky 2D - needs proper implementation
      int nBlkRow= (rowLenF + VT_BLKM) / VT_BLKN;
      int rowStride= BITS_TO_WRDSH(rowLenF,5); //(rowLenF + 31) / 32;
      for (int i=0; i<nRows; i++)
      {
         vThreshL32<<<nBlkRow,VT_BLKN>>>(pBM + i * rowStride, pF + i * rowLenF, rowLenF, *pMC);
      }
      r= (0 == ctuErr(NULL, "nRows*vThreshL32()"));
   }
   LOG("binMapCudaRowsF32(.., rowLen=%d, BM(%f,0x%X) ) - dt= %Gms\n", rowLenF, pMC->t[0], pMC->m, t.elapsedms());
   //cudaDeviceSynchronize(); // stream sync provided by timer
   return(r);
} // binMapCudaRowsF32

static int planarity (const int def[3], const Stride s[3])
{  // Only works for full image def, not sub region!
   return((1 == s[0]) + (def[0] == s[1]) + (def[0] * def[1] == s[2]));
} // planar

extern "C"
BMStrideDesc *binMapCUDA
(
   uint        * pBM,
   BMStrideDesc * pBMSD,
   const MultiFieldInfo * pMFI,
   const BinMapF32      * pMC
)
{
   const size_t nBM= setBMSD(pBMSD, pMFI->def, pMFI->profile); //(planeStride * def[2])

   if (1 == pMFI->nField)
   {
       if ((0 == (pMFI->def[0] & 0x1F)) &&
            (3 == planarity(pMFI->def, pMFI->mfd.stride.s)))
      {  // Treat as 1D - probably fastest?
         const size_t nF= prodNI(pMFI->def,3);
         vThreshL32<<<nF/VT_BLKN,VT_BLKN>>>(pBM, pMFI->mfd.pF32[0], nF, *pMC);
         if (0 == ctuErr(NULL, "vThreshL32()")) { return(pBMSD); }
      }
      else
      {
         int nBlkRow= (pMFI->def[0] + VT_BLKM) / VT_BLKN;
         dim3 grid(nBlkRow, pMFI->def[1], pMFI->def[2]);
         vThreshV32<<<grid,dim3(VT_BLKN,1,1)>>>(pBM, pMFI->mfd.pF32[0], pMFI->def[0], *pBMSD, pMFI->mfd.stride, *pMC);
         if (0 == ctuErr(NULL, "vThreshV32()")) { return(pBMSD); }
      }
   }
   else
   {
      //vThreshSum<<<nBlk,VT_BLKN>>>(pBM, pMFI->mfd, pMFI->nField, nF, *pMC);

      //return(pBMSD);
   }
   return(NULL);
} // binMapCUDA
