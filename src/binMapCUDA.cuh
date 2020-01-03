// binMapCUDA.cuh - Templated device code for Packed Binary Map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan 2019 - Jan 2020

#ifndef BIN_MAP_CUDA_CUH
#define BIN_MAP_CUDA_CUH

#ifdef NO_WLP
#define MERGE32_SAFE
#endif

// Hacky definitions for warp-block setup
#define VT_WRDS 5
#define VT_WRDN (1<<VT_WRDS)
#define VT_WRDM (VT_WRDN-1)

#define VT_BLKS 7             // 7
#define VT_BLKN (1<<VT_BLKS)
#define VT_BLKM (VT_BLKN-1)

#define VT_BWS (VT_BLKS - VT_WRDS)

#define CAT_NUM 2


/******************** Device classes ********************/

class BinMapCUDAOrg
{
protected:
   FieldStride fs[3];
   uint  rowElem;
   BMStride rowWS, planeWS;   // 32b word strides

   bool setDS (const FieldStride *pS, const FieldDef *pD)
   {
      if (pS)
      {  // copy - check >= def ?
         for (int i=0; i<3; i++) { fs[i]= pS[i]; }
      }
      else { return(genStride(fs, 3, 0, pD, 1) > 0); }
      return(false);
   } // setDS

public:
   BinMapCUDAOrg (const BMOrg *pO, const BMFieldInfo *pI)
   {
      setDS(pI->pS, pI->pD);
         //copyOrGenStride(stepF, 1, 2, pI->pD, pI->pS);
      rowElem= pO->rowElem;
      rowWS=   pO->rowWS;
      planeWS= pO->planeWS;
   } // CTOR

   FieldStride planeStrideField (void) const { return(fs[2]); }
   BMStride planeStrideBM (void) const { return(planeWS); }

   __device__ bool inRow (uint x) const { return(x < rowElem); }
   __device__ size_t fieldIndex (uint x, uint y, uint z) const { return(x * fs[0] + y * fs[1] + z * fs[2]); }
   __device__ size_t bmIndex (uint x, uint y, uint z) const { return((x >> VT_WRDS) + y * rowWS + z * planeWS); }

}; // BinMapCUDAOrg


/********** Templated device class hierrachy **********/

template <typename T_Elem>
class CUDAImgMom
{
protected:
   T_Elem * pM2;

   CUDAImgMom (void *p) { pM2= static_cast<T_Elem*>(p); }

   // Block-wide sum reduction
   __device__ void sumRB (T_Elem *pR, const T_Elem v)
   {
      __shared__ T_Elem t[VT_BLKN];
      t[threadIdx.x]= v;
      //#pragma unroll 5
      for (int s= blockDim.x>>1; s > 0; s>>= 1)
      {  __syncthreads(); // Keep block together
         if (threadIdx.x < s) { t[threadIdx.x]+= t[threadIdx.x+s]; }
      }
      if (0 == threadIdx.x) { atomicAdd( pR, t[0] ); }
      __syncthreads();
   } // sumRB

   // Block-wide categorical sum reduction
   __device__ void sumCatRB (T_Elem *pR, const int strideR, const T_Elem v, const int c)
   {
      __shared__ T_Elem t[CAT_NUM][VT_BLKN];
#if CAT_NUM > 2
      #pragma unroll CAT_NUM
      for (int i=0; i<CAT_NUM; i++) { t[i][threadIdx.x]= 0; }
#else
      t[c^0x1][threadIdx.x]= 0;
#endif
      t[c][threadIdx.x]= v;
      for (int s= blockDim.x>>1; s > 0; s>>= 1)
      {  __syncthreads(); // Keep block together
         if (threadIdx.x < s)
         {
            #pragma unroll CAT_NUM
            for (int i=0; i<CAT_NUM; i++) { t[i][threadIdx.x]+= t[i][threadIdx.x+s]; }
         }
      }
      if (0 == threadIdx.x)
      {
         #pragma unroll CAT_NUM
         for (int i=0; i<CAT_NUM; i++) { atomicAdd( pR+i*strideR, t[i][0] ); }
      }
      __syncthreads();
   } // sumCatRB

   __device__ void add (T_Elem m)
   {
      sumRB(m);
      sumRB(m*m);
   } // add

   __device__ void addCat (T_Elem m, uint c)
   {
      sumCatRB(pM2+0, 2, m, c);
      sumCatRB(pM2+1, 2, m*m, c);
   } // addCat

}; // class CUDAImgMom

template <typename T_Elem>
class CUDAMap
{
protected:
   T_Elem   t[BM_NUMT];
   uint     m;

public:
   CUDAMap (const BinMapF32 *pM) { m= pM->m; for (int i=0; i<BM_NUMT; i++) { t[i]= pM->t[i]; } }
   CUDAMap (const BinMapF64 *pM) { m= pM->m; for (int i=0; i<BM_NUMT; i++) { t[i]= pM->t[i]; } }

   __device__ uint eval (const T_Elem f) const
   {
      const uint d= (1 - (f < t[BM_TLB]) + (f > t[BM_TUB])); // t[1]
      return( (m >> d) & 0x1 );
   } // eval
   //__device__ uint operator () (const T_Elem f) const { eval(f); }
}; // template class CUDAMap

template <typename T_Elem>
class CUDAFieldMap : protected CUDAMap<T_Elem>
{
protected:
   const T_Elem * pF;

public:                 // static_cast< const T_Elem * >() - irrelevant so why bother?
   size_t iOff; // Hacky debug/test aid


   CUDAFieldMap (const BMFieldInfo *pI, const BinMapF32 *pM) : CUDAMap<T_Elem>(pM)
   { copyValidPtrByMask( (ConstFieldPtr*)&pF, 1, pI->pFieldDevPtrTable, pI->fieldTableMask); iOff= 0; }
   CUDAFieldMap (const BMFieldInfo *pI, const BinMapF64 *pM) : CUDAMap<T_Elem>(pM)
   { copyValidPtrByMask( (ConstFieldPtr*)&pF, 1, pI->pFieldDevPtrTable, pI->fieldTableMask); iOff= 0; }

   //__device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::operator () (pF[i]); }
   __device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::eval(pF[i]); }

   void setOffset (size_t i=0)
   {
      //const T_Elem *p= pF;
      pF+= (signed)i - (signed)iOff;
      //if (log) { LOG("CUDAFieldMap::setOffset(%zu) %p->%p\n", i, p, pF); }
      iOff= i;
   } // setOffset
   void addOffset (size_t i=0) { pF+= i; iOff+= i; }

}; // template class CUDAFieldMap

template <typename T_Elem>
class CUDAMultiField
{  // Multiple fields with common stride
protected:
   const T_Elem   * fPtrTab[BMCU_FIELD_MAX];
   uint           nF;

public:
   CUDAMultiField (const BMFieldInfo *pI)
   {
      if (pI->pD)
      {
         nF= copyValidPtrByMask( (ConstFieldPtr*)fPtrTab, BMCU_FIELD_MAX, pI->pFieldDevPtrTable, pI->fieldTableMask);
      }
   } // CTOR

   __device__ T_Elem sum (const size_t i) const
   {
      T_Elem s= (fPtrTab[0])[i];
      for (int iF=1; iF < nF; iF++) { s+= (fPtrTab[iF])[i]; }
      return(s);
   } // sum
   //__device__ T_Elem operator () (const size_t i) const { return sum(i); }
}; // template class CUDAMultiField

// TODO: polymorphisation of CUDAFieldMap & CUDAMultiFieldMap ... ?
template <typename T_Elem>
class CUDAMultiFieldMap : protected CUDAMultiField<T_Elem>, CUDAMap<T_Elem>
{
public:
   CUDAMultiFieldMap (const BMFieldInfo *pI, const BinMapF32 *pM) : CUDAMultiField<T_Elem>(pI), CUDAMap<T_Elem>(pM) {;}
   CUDAMultiFieldMap (const BMFieldInfo *pI, const BinMapF64 *pM) : CUDAMultiField<T_Elem>(pI), CUDAMap<T_Elem>(pM) {;}

   __device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::eval( CUDAMultiField<T_Elem>::sum(i) ); }
}; // template class CUDAMultiFieldMap


/* Device utility functions */
//#define NO_WLP
//#define MERGE32_SAFE

#ifdef NO_WLP

__device__ uint mergeOR32 (volatile uint w[32], const int lane)
{
#ifdef MERGE32_SAFE
   for (int s= 16; s > 0; s>>= 1)
   {  __syncthreads(); // Slightly slower but keeps warp together
      if (lane < s) { w[lane]|= w[lane+s]; }
   }
#else
   if (lane < 16) // DANGER! Half warp will free-run to next sync -
   {  // DO NOT USE when warp-wide functions are subsequently employed!
      #pragma unroll 5 // Ineffective? Presume limited by shared read/write
      for (int s= 16; s > 0; s>>= 1) { __syncthreads(); w[lane]|= w[lane+s]; }
   }
#endif
   return(w[lane]);
} // mergeOR32

__device__ uint bitMergeShared (uint v)
{
   __shared__ BMPackWord w[VT_BLKN]; // Caution! fixed blocksize!

   w[threadIdx.x]= v;
   return mergeOR32(w + (threadIdx.x & ~VT_WRDM), threadIdx.x & VT_WRDM);
} // bitMergeShared

#define BIT_MERGE(w) bitMergeShared(w)

#else // NO_WLP

__device__ uint bitMergeBallot (uint v)
{  // Nice simplification but doesn't work... ?
   return __ballot_sync(VT_WRDM, v);
} // bitMergeBallot

#define SHFL_MASK_ALL (-1) // (1<<warpSize)-1 or 0xFFFFFFFF

// TODO - FIX - warning: integer conversion resulted in a change of sign
// #pragma NVCC warning disable ?

__device__ uint bitMergeWarp (uint w)
{  // CUDA9 warp level primitives (supported on CUDA7+ ? )
#if 1
   #pragma unroll 5 // Seems to have no effect - need to enable a setting ?
   for (int s= warpSize/2; s > 0; s>>= 1) { w|= __shfl_down_sync(SHFL_MASK_ALL, w, s); }
#else // Manual unroll is faster (but reducing mask in successive steps makes negligible difference)
   w|= __shfl_down_sync(SHFL_MASK_ALL, w, 16);
   w|= __shfl_down_sync(SHFL_MASK_ALL, w, 8);
   w|= __shfl_down_sync(SHFL_MASK_ALL, w, 4);
   w|= __shfl_down_sync(SHFL_MASK_ALL, w, 2);
   w|= __shfl_down_sync(SHFL_MASK_ALL, w, 1);
#endif
   return(w);
} // bitMergeWarp

#define BIT_MERGE(w) bitMergeWarp(w)

#endif // NO_WLP

/* KERNELS */

//<T_Map>
template <typename T_Elem>
__global__ void mapField (BMPackWord rBM[], const CUDAFieldMap<T_Elem> f, const size_t n)
{  //
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n)
   {
      const uint j= threadIdx.x & VT_WRDM;   // lane index (bit number)

      uint w= BIT_MERGE( f(i) << j );

      if (0 == j) { rBM[i>>VT_WRDS]= w; }
   }
} // mapField

template <typename T_Elem>
__global__ void mapMultiField (BMPackWord rBM[], const CUDAMultiFieldMap<T_Elem> mf, const size_t n)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n)
   {
      const uint j= threadIdx.x & VT_WRDM;   // lane index (bit number)

      uint w= BIT_MERGE( mf(i) << j );

      if (0 == j) { rBM[i>>VT_WRDS]= w; }
   }
} // mapMultiField

template <typename T_Elem>
__global__ void mapStrideField (BMPackWord rBM[], const BinMapCUDAOrg org, const CUDAFieldMap<T_Elem> f)
{
   const uint x= blockIdx.x * blockDim.x + threadIdx.x;
   if (org.inRow(x))
   {
      size_t i= org.fieldIndex(x, blockIdx.y, blockIdx.z);
      const uint j= threadIdx.x & VT_WRDM;   // lane index (bit number)

      uint w= BIT_MERGE( f(i) << j );

      if (0 == j) { rBM[org.bmIndex(x, blockIdx.y, blockIdx.z)]= w; }
   }
} // mapStrideField

template <typename T_Elem>
__global__ void mapStrideMultiField (BMPackWord rBM[], const BinMapCUDAOrg org, const CUDAMultiFieldMap<T_Elem> mf) // const size_t n)
{
   const uint x= blockIdx.x * blockDim.x + threadIdx.x;
   if (org.inRow(x))
   {
      size_t i= org.fieldIndex(x, blockIdx.y, blockIdx.z);
      const uint j= threadIdx.x & VT_WRDM;

      uint w= BIT_MERGE( mf(i) << j );

      if (0 == j) { rBM[org.bmIndex(x, blockIdx.y, blockIdx.z)]= w; }
   }
} // mapStrideMultiField

#endif // BIN_MAP_CUDA_CUH
