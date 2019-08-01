// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#include "binMapCUDA.h"


#define VT_BLKS 5
#define VT_BLKD (1<<VT_BLKS)
#define VT_BLKM (VT_BLKD-1)

/***/

__device__ int bm1f32 (const float f, const BinMapF32& bm)
{
   const int d= (1 + (f > bm.t[0]) - (f < bm.t[0]));
   return( (bm.m >> d) & 0x1 );
} // bm1f32

__global__ void vThresh8 (uint r[], const float f[], const size_t n, const BinMapF32 mc)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[VT_BLKD];
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

__global__ void vThresh32 (uint r[], const float f[], const size_t n, const BinMapF32 bm)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[VT_BLKD];
   if (i < n)
   {
      const int j= i & VT_BLKM;

      z[j]= bm1f32(f[i],bm) << j;

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

extern "C"
void binMapCudaRowsF32
(
   U32 * pBM,
   const F32 * pF,
   const int rowLenF,      // row length ie. "X dimension"
   const int rowStrideBM,  // 32bit word stride of rows of packed binary map, should be >=
   const int nRows,        // product of other "dimensions" (Y * Z)
   const BinMapF32 *pMC
)
{
   int blkD= VT_BLKD;
   LOG("***\binMapCudaRowsF32() - bmc: %f,0x%X\n",pMC->t[0], pMC->m);
   if (0 == (rowLenF & 0x1F))
   {
      size_t nF= rowLenF * nRows;
      int nBlk= (nF + blkD-1) / blkD;
      // CAVEAT! Treated as 1D
      vThresh32<<<nBlk,blkD>>>(pBM, pF, nF, *pMC);
   }
   else
   {
      int rowStride= BITS_TO_WRDSH(rowLenF,5); //(rowLenF + 31) / 32;
      int nBlk= (rowLenF + blkD-1) / blkD;
      for (int i=0; i<nRows; i++)
      {
         vThresh32<<<nBlk,blkD>>>(pBM + i * rowStride, pF + i * rowLenF, rowLenF, *pMC);
      }
   }
   ctuErr(NULL, "vThresh32()");
   cudaDeviceSynchronize();
} // binMapCudaRowsF32

