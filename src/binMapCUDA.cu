// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Sept 2019

#include "binMapCUDA.h"
#include "utilCUDA.hpp"


#define VT_WRDS 5
#define VT_WRDN (1<<VT_WRDS)
#define VT_WRDM (VT_WRDN-1)

#define VT_BLKS 7             // 7
#define VT_BLKN (1<<VT_BLKS)
#define VT_BLKM (VT_BLKN-1)

#define VT_BWS (VT_BLKS - VT_WRDS)

#define CAT_NUM 2

/* Misc functions */



/* Host side utility class/struct */
     *pD;

int planarity (const FieldDef def[], const FieldStride stride[], int n)
{
   int i=0, r=0;
   if (n > 0)
   {
      FieldStride s= 1;
      do
      {
         r+= (s == stride[i])
         s*= def[i];
      } while (++i < n);
   }
   return(r);
} // planarity

struct Region
{  // Expect multiple fields, common def & stride
   size_t   nElem;
   FieldDef elemDef[3];
   int      grdDef0, blkDef0;
   uint8_t  nD, nF, nS, flags;

#define REG_AS1D (0x01)
//public:
   bool validate (const BMFieldInfo *pI) // TODO: SubRegion/Box ???
   {
      if (pI)
      {
         nD= 0; nF= 0;
         if (pI->pFieldDevPtrTable)
         {
            nF= countValidPtrByMask(pI->pFieldDevPtrTable, pI->fieldTableMask);
         }
         if (pI->pD)
         {
            for (int i=0; i<3; i++)
            {
               elemDef[nD]= pI->pD[i];
               nD+= (elemDef[nD] > 1);
            }
            for (int i=nD; i<3; i++) { elemDef[i]= 1; }
            blkDef0= VT_BLKN;
            //if (blkDim0...
            grdDef0= (elemDef[0] + (blkDef0-1) ) / blkDef0;
            nElem= prodNI(elemDef,3);
            if ( (elemDef[0] == nElem) ||
               ( (0 == (elemDef[0] & VT_WRDM)) &&
                  ((NULL == pD->pS) || (3 == planarity(elemDef, pI->pS, 3)) ) ) )
            {
               flags|= REG_AS1D;
            }
         }
         return((nD > 0) && (nF > 0));
      }
      return(false);
   } // validate (as conditional 'CTOR')

   bool collapsable (void) const { return(flags & REG_AS1D); }

   // 3D vs. 1D-collapsed launch params
   dim3 blkDef (void) const { return dim3(blkDef0, 1, 1); }
   dim3 grdDef (void) const { return dim3(grdDef0, elemDef[1], elemDef[2]); }
   int blkDefColl (void) const { return(blkDef0); }
   int grdDefColl (void) const { if (blkDef0 > 0) { return((nElem + blkDef0-1) / blkDef0); } else return(0); }
   // Methods for hacky test code
   int blkDefRow (void) const { return(blkDef0); }
   int grdDefRow (void) const { return(grdDef0); }
   size_t blkShMem (void) { return( blkDef0 * sizeof(uint) ); }
}; // struct Region


/* Device classes */

class CUDAOrg
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
   CUDAOrg (const BMOrg *pO, const BMFieldInfo *pI)
   {
      setDS(pI->pS, pI->pD);
      rowElem= pO->rowElem;
      rowWS=   pO->rowWS;
      planeWS= pO->planeWS;
   } // CTOR

   __device__ bool inRow (uint x) const { return(x < rowElem); }
   __device__ size_t fieldIndex (uint x, uint y, uint z) const { return(x * fs[0] + y * fs[1] + z * fs[2]); }
   __device__ size_t bmIndex (uint x, uint y, uint z) const { return((x >> VT_WRDS) + y * rowWS + z * planeWS); }

}; // CUDAOrg

/* Templated device classes */

template <typename T_Elem>
class CUDAImgMom
{
protected:
   T_Elem * pM2;

   CUDAImgMom (void *p) { pM2= static_cast<T_Elem*>(p); }

   __device__ void sum (T_Elem *pR, const T_Elem v) // Block-wide sum reduction
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
   } // sum

   __device__ void sumCat (T_Elem *pR, const int strideR, const T_Elem v, const int c) // Block-wide categorical sum reduction
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
   } // sumCat

   __device__ void add (int x, int y, int z, T_Elem m)
   {
      T_Elem s, m2= m * m;

      s= sum(m);
      if (0 == threadIdx.x) { atomicAdd( pM2+0, s); }
      s= sum(m2);
      if (0 == threadIdx.x) { atomicAdd( pM2+1, s); }
   }
   __device__ void addCat (int x, int y, int z, T_Elem m, uint c)
   {
      sumCat(pM2+0, 2, m, c);
      sumCat(pM2+1, 2, m*m, c);
   }

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
      const uint d= (1 + (f > t[0]) - (f < t[0]));
      return( (m >> d) & 0x1 );
   } // eval
   __device__ uint operator () (const T_Elem f) const { eval(f); }
}; // template class CUDAMap

template <typename T_Elem>
class CUDAFieldMap : protected CUDAMap<T_Elem>
{
protected:
   const T_Elem * pF;
   size_t iOff; // Hacky debug/test aid

public:                 // static_cast< const T_Elem * >() - irrelevant so why bother?
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
   const T_Elem   * fPtrTab[BMFI_FIELD_MAX];
   uint           nF;

public:
   CUDAMultiField (const BMFieldInfo *pI)
   {
      if (pI->pD)
      {
         nF= copyValidPtrByMask( (ConstFieldPtr*)fPtrTab, BMFI_FIELD_MAX, pI->pFieldDevPtrTable, pI->fieldTableMask);
      }
   } // CTOR

   //__device__ size_t index (uint x, uint y, uint z) const { return(x * stride[0] + y * stride[1] + z * stride[2]); }

   __device__ T_Elem sum (const size_t i) const
   {
      T_Elem s= (fPtrTab[0])[i];
      for (int iF=1; iF < nF; iF++) { s+= (fPtrTab[iF])[i]; }
      return(s);
   } // sum
   __device__ T_Elem operator () (const size_t i) const { return sum(i); }
}; // template class CUDAMultiField

// TODO: polymorphisation of CUDAFieldMap & CUDAMultiFieldMap ... ?
template <typename T_Elem>
class CUDAMultiFieldMap : protected CUDAMultiField<T_Elem>, CUDAMap<T_Elem>
{
public:
   CUDAMultiFieldMap (const BMFieldInfo *pI, const BinMapF32 *pM) : CUDAMultiField<T_Elem>(pI), CUDAMap<T_Elem>(pM) {;}
   CUDAMultiFieldMap (const BMFieldInfo *pI, const BinMapF64 *pM) : CUDAMultiField<T_Elem>(pI), CUDAMap<T_Elem>(pM) {;}

   //__device__ size_t index (uint x, uint y, uint z) const { return CUDAMultiField<T_Elem>::index(x,y,z); }
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

template <typename T_Elem>
__global__ void mapField (BMPackWord rBM[], const CUDAFieldMap<T_Elem> f, const size_t n)
{
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
__global__ void mapStrideField (BMPackWord rBM[], const CUDAOrg org, const CUDAFieldMap<T_Elem> f)
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
__global__ void mapStrideMultiField (BMPackWord rBM[], const CUDAOrg org, const CUDAMultiFieldMap<T_Elem> mf) // const size_t n)
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


/* INTERFACE */

extern "C"
BMOrg *binMapCUDA
(
   BMPackWord  * pW,
   BMOrg       * pO,
   const BMFieldInfo * pI,
   const BinMapF32   * pM
)
{
   Region reg;

   //if (32 != warpSize) { WARN("[binMapCUDA] warpSize=%d\n", warpSize); }
   if (reg.validate(pI))
   {
      CTimerCUDA t;
      const char * pID= NULL;

      if ( setBMO(pO, reg.elemDef, pI->profID) )
      {
         //LOG("Region::validate() - D%d F%d\n", reg.nD, reg.nF);
         switch (pI->profID & 0x70)
         {
            case 0x00 :
            if (reg.collapsable() && (1 == reg.nF))
            {
               if (reg.nS > 1)
               {
                  CUDAStrmBlk s;
                  //FieldStride stride; genStride(&stride, 1, 2, pI->pD, 1); // LOG("rowStride=%d\n", rowStride);
                  CUDAFieldMap<float> map(pI, pM);
                  const int blkStrm= reg.blkDefColl();
                  const int grdStrm= reg.grdDefColl() / reg.nS;
                  const FieldStride stride= blkStrm * grdStrm;
                  const int wStride= stride / 32;
                  const int resid= reg.grdDefColl() - (nStrm * grdStrm);
                  int i, n= reg.nS - (resid > 0);
                  for (i=0; i<n; i++)
                  {
                     mapField<<< grdStrm, blkStrm, 0, s[i] >>>(pW+(i*wStride), map, stride);
                     ctuErr(NULL, "mapField");
                     map.addOffset(stride);
                     //map.setOffset((1+i) * stride);
                  }
                  if (resid > 0)
                  {
                     mapField<<< grdStrm+resid, blkStrm, 0, s[i] >>>(pW+(i*wStride), map, reg.nElem-(i*stride));
                     ctuErr(NULL, "mapField (resid)");
                  }
                  pID= "nStrm*mapField()";
               }
               else
               {
                  mapField<<< reg.grdDefColl(), reg.blkDefColl() >>>(pW, CUDAFieldMap<float>(pI, pM), reg.nElem);
                  pID= "mapField()";
               }
               break;
            } // else...
            case 0x10 :
            if (1 == reg.nF)
            {
               mapStrideField<<< reg.grdDef(), reg.blkDef() >>>(pW, CUDAOrg(pO, pI), CUDAFieldMap<float>(pI, pM));
               pID= "mapStrideField()";
               break;
            } // else...
            case 0x20 :
            if (reg.collapsable())
            {
               mapMultiField<<< reg.grdDefColl(), reg.blkDefColl() >>>(pW, CUDAMultiFieldMap<float>(pI, pM), reg.nElem);
               pID= "mapMultiField()";
               break;
            } // else ...
            case 0x30 :
               mapStrideMultiField<<< reg.grdDef(), reg.blkDef() >>>(pW, CUDAOrg(pO, pI), CUDAMultiFieldMap<float>(pI, pM));
               pID= "mapStrideMultiField()";
               break;
            case 0x40 : // HACK! ALERT! flaky test code!
            {  const int nRows= prodNI(reg.elemDef+1,2);
               CUDAFieldMap<float> map(pI, pM);
               FieldStride rowStride; genStride(&rowStride, 1, 1, pI->pD, 1); // LOG("rowStride=%d\n", rowStride);
               CUDAStrmBlk s;

               //LOG("\tinit dt=%Gms\n", t.elapsedms());
               //t.stampStream(s[0]);

               for (int i=0; i<nRows; i++)
               {  // Horribly inefficient single row iteration
                  map.setOffset(i * rowStride); // i>(nRows-10));
                  mapField<<< reg.grdDefRow(), reg.blkDefRow(), 0, s[i] >>>(pW + i * pO->rowWS, map, reg.elemDef[0]);
               }
               //LOG("\tsubmit dt=%Gms\n", t.elapsedms(CDTM_AF_STAMP|CDTM_AF_SYNC, s[0]));

               pID= "nRows*mapField()"; break;
            }
         }
         LOG("binMapCUDA() - %s<<<%u>>>() - dt= %Gms\n", pID, reg.blkDefColl(), t.elapsedms());
         if (0 == ctuErr(NULL, pID)) { return(pO); }
      }
   }
   return(NULL);
} // binMapCUDA
