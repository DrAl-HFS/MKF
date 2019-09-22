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

struct CUDAFieldDesc
{  // Expect multiple fields, common def & stride
   int            nF;
   FieldDef       def[3];
   FieldStride    stride[3];
   ConstFieldPtr  field[BMFI_FIELD_MAX];
};

// Refactor to CTOR + check ? ...
static int checkFD (CUDAFieldDesc *pD, const BMFieldInfo *pI)
{
   if (pD && pI && pI->pD)
   {
      int pad= (pI->pD[0] & VT_WRDM);
      int n=0, m= MIN(BMFI_FIELD_MAX, pI->nField);
      if (m > 0)
      {
         for (int i=0; i<m; i++)
         {
            pD->field[n]= pI->field[i];
            n+= (NULL != pI->field[i].p);
         }
         //if (n != m) WARN();
         pD->nF= n;
         if (pI->pS)
         {  // Validate stride ?
            for (int i=0; i<3; i++) { pD->def[i]= pI->pD[i]; pD->stride[i]= pI->pS[i]; }
         }
         else
         {  // Generate stride
            FieldStride k= 1;
            for (int i=0; i<3; i++)
            {
               pD->def[i]= pI->pD[i];
               pD->stride[i]= k;
               k*= pD->def[i];
            }
         }
         if (n > 1) { return(4); } // if (pad) WARN();
         else { return( 1 + (0 != pad) ); }
      }
   }
   return(0);
} // checkFD

/***/

template <typename T_Elem>
class CUDAMap
{
protected:
   T_Elem t[BM_NUMT];
   uint     m;

public:
   CUDAMap (const BinMapF32 *pM) { m= pM->m; for (int i=0; i<BM_NUMT; i++) { t[i]= pM->t[i]; } }
   CUDAMap (const BinMapF64 *pM) { m= pM->m; for (int i=0; i<BM_NUMT; i++) { t[i]= pM->t[i]; } }

   __device__ uint eval (const T_Elem f) const
   {
      const uint d= (1 + (f > t[0]) - (f < t[0]));
      return( (m >> d) & 0x1 );
   } // eval
   __device__ uint operator () (const T_Elem f) const { eval(f); }
}; // template class CUDAMap

template <typename T_Elem>
class CUDAFieldMap : protected CUDAMap<T_Elem>
{
protected:
   const T_Elem *pF;

public:
   CUDAFieldMap (const T_Elem * p, const BinMapF32 *pM) : CUDAMap<T_Elem>(pM) { pF= p; }
   CUDAFieldMap (const T_Elem * p, const BinMapF64 *pM) : CUDAMap<T_Elem>(pM) { pF= p; }

   //__device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::operator () (pF[i]); }
   __device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::eval(pF[i]); }
}; // template class CUDAFieldMap

#if 0

template <typename T_Elem>
class CUDAMultiField
{  // Multiple fields with common stride
protected:
   const T_Elem   * fPtr[BMFI_FIELD_MAX];
   FieldStride stride[3];
   FieldDef    def[3];
   uint        nF;

   void setDS (const FieldDef d[3], const FieldStride *pS)
   {
      if (pS)
      {  // copy
         for (int i=0; i<3; i++) { def[i]= d[i]; stride[i]= pS[i]; }
      }
      else
      {  // Generate stride
         FieldStride k= 1;
         for (int i=0; i<3; i++)
         {
            def[i]= d[i];
            stride[i]= k;
            k*= def[i];
         }
      }
   } // setDS

   uint setF (ConstFieldPtr a[], const uint m)
   {
      uint n= 0;
      for (int i=0; i<BMFI_FIELD_MAX; i++)
      {
         if (m & (0x1 << i)) { fPtr[n]= a[i].p; }
         n+= (NULL != fPtr[i]);
      }
      return(n);
   } // setF

public:
   CUDAMultiField (const BMFieldInfo *pI)
   {
      if (pI->pD)
      {
         setDS(pI->pD, pI->pS);
         nF= setF(pI->field, 0xF);
      }
   } // CTOR

   __device__ T_Elem operator () (const size_t i) const
   {
      T_Elem s= (fPtr[0])[i];
      for (int iF=1; iF < nF; iF++) { s+= (fPtr[iF])[i]; }
      return(s);
   } // operator ()
}; // template class CUDAMultiField

template <typename T_Elem>
class CUDAMultiFieldMap : protected CUDAMultiField<T_Elem>, CUDAMap<T_Elem>
{
public:
   CUDAMultiFieldMap (const BMFieldInfo *pI, const BinMapF32 *pM) : CUDAMultiField<T_Elem>(pI), CUDAMap<T_Elem>(pM) {;}
   CUDAMultiFieldMap (const BMFieldInfo *pI, const BinMapF64 *pM) : CUDAMultiField<T_Elem>(pI), CUDAMap<T_Elem>(pM) {;}

   __device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::( CUDAMultiField<T_Elem>::(i) ); }
}; // template class CUDAMultiFieldMap

#endif

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

__device__ int bm1f32 (const float f, const BinMapF32& bm)
{
   const int d= (1 + (f > bm.t[0]) - (f < bm.t[0]));
   return( (bm.m >> d) & 0x1 );
} // bm1f32

__device__ void merge32 (BMPackWord u[32], const int j)
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

template <typename T_Elem>
__global__ void mapFieldL32 (BMPackWord r[], const CUDAFieldMap<T_Elem> f, const size_t n)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint z[VT_BLKN];
   if (i < n)
   {
      const int j= i & VT_BLKM;

      z[j]= f(i) << j;

      merge32(z, j);
      if (0 == j) { r[i>>VT_WRDS]= z[0] | z[16]; }
   }
} // mapFieldL32

__global__ void vThreshL32 (BMPackWord r[], const float f[], const size_t n, const BinMapF32 bm)
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

__global__ void vThreshV32
(
   BMPackWord rBM[],
   const CUDAFieldDesc fd,
   const BMOrg bmo,
   const BinMapF32 bm
)
{
   const int x= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint u[VT_BLKN];
   if (x < fd.def[0])
   {
      size_t i= x * fd.stride[0] + blockIdx.y * fd.stride[1] + blockIdx.z * fd.stride[2];
      const int j= i & VT_BLKM;

      u[j]= bm1f32( fd.field[0].pF32[i], bm ) << j; // (j & VT_WRDM)

      merge32(u, j);
      if (0 == j) // & VT_WRDM) if BLKS > WRDS !
      {  // (x >> VT_WRDS)
         i= blockIdx.x + blockIdx.y * bmo.rowWS + blockIdx.z * bmo.planeWS;
         rBM[i]= u[0] | u[16];
      }
   }
} // vThreshV32

__global__ void vThreshVSum32
(
   BMPackWord rBM[],
   const CUDAFieldDesc fd,
   const BMOrg bmo,
   const BinMapF32 bm
)
{
   const int x= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ uint u[VT_BLKN];
   if (x < fd.def[0])// && (y < defY))
   {
      size_t i= x * fd.stride[0] + blockIdx.y * fd.stride[1] + blockIdx.z * fd.stride[2];
      const int j= i & VT_BLKM;

      float s= (fd.field[0].pF32)[i];
      for (int f=1; f < fd.nF; f++) { s+= fd.field[f].pF32[i]; }

      u[j]= bm1f32(s,bm) << j;

      merge32(u, j);
      if (0 == j)
      {
         i= (x >> VT_WRDS) + blockIdx.y * bmo.rowWS + blockIdx.z * bmo.planeWS;
         rBM[i]= u[0] | u[16];
      }
   }
} // vThreshVSum32


// DEPRECATED
static int binMapCudaRowsF32
(
   BMPackWord * pBM,
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
      for (int i=0; i<nRows; i++)
      {
         vThreshL32<<<nBlkRow,VT_BLKN>>>(pBM + i * rowStrideBM, pF + i * rowLenF, rowLenF, *pMC);
      }
      r= (0 == ctuErr(NULL, "nRows*vThreshL32()"));
   }
   LOG("binMapCudaRowsF32(.., L=%d, S=%d, N=%d, BM(%f,0x%X) ) - dt= %Gms\n", rowLenF, rowStrideBM, nRows, pMC->t[0], pMC->m, t.elapsedms());
   //cudaDeviceSynchronize(); // stream sync provided by timer
   return(r);
} // binMapCudaRowsF32


/* INTERFACE */

extern "C"
BMOrg *binMapCUDA
(
   BMPackWord  * pW,
   BMOrg       * pO,
   const BMFieldInfo * pF,
   const BinMapF32   * pMC
)
{
   CUDAFieldDesc fd;
   const int id= checkFD(&fd, pF);
   if (id > 0)
   {
      CTimerCUDA t;
      const char * pID= NULL;
      const int   nBlkRow= (fd.def[0] + VT_BLKM) / VT_BLKN;
      setBMO(pO, fd.def, pF->profile);
      if (id <= 2)
      {
#if 0
         binMapCudaRowsF32(pW, fd.field[0].pF32, fd.def[0], pO->rowWS, prodNI(fd.def+1,2), pMC);
         pID= "binMapCudaRowsF32()";
#else
         switch (id)
         {
            case 1 :
            {  const size_t nF= prodNI(fd.def,3);
               mapFieldL32<<<nF/VT_BLKN,VT_BLKN>>>(pW, CUDAFieldMap<float>(fd.field[0].pF32, pMC), nF);
//               vThreshL32<<<nF/VT_BLKN,VT_BLKN>>>(pW, fd.field[0].pF32, nF, *pMC);
               pID= "map()"; // "vThreshL32()";
            }  break;
            case 2 : // Horribly inefficient iteration - only method presently working for !=*32 row length
            {  const int nRows= prodNI(fd.def+1,2);
               for (int i=0; i<nRows; i++)
               {
                  vThreshL32<<<nBlkRow,VT_BLKN>>>(pW + i * pO->rowWS, fd.field[0].pF32 + i * fd.stride[1], fd.def[0], *pMC);
               }
               pID= "nRows*vThreshL32()";
            }  break;
         }
#endif
      }
      else
      {
         const dim3 grd(nBlkRow, fd.def[1], fd.def[2]);
         const dim3 blk(VT_BLKN,1,1);
         switch (id)
         {
            case 3 :
               vThreshV32<<<grd,blk>>>(pW, fd, *pO, *pMC);
               pID= "vThreshV32()";
               break;
            case 4 :
               vThreshVSum32<<<grd,blk>>>(pW, fd, *pO, *pMC);
               pID= "vThreshVSum32()";
               break;
         }
      }
      LOG("binMapCUDA() - %s - dt= %Gms\n", pID, t.elapsedms());
      if (0 == ctuErr(NULL, pID)) { return(pO); }
   }
   return(NULL);
} // binMapCUDA
