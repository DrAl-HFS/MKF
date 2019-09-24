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


/***/

uint copyValidPtrByMask (ConstFieldPtr r[], const int max, const ConstFieldPtr a[], const uint mask)
{
   uint t, i= 0, n= 0;
   do
   {
      t= (0x1 << i);
      if ((NULL != a[i].p) && (mask & t))
      {
         r[n++]= a[i];
      }
      i++;
   } while ((mask > t) && (n < max));
   return(n);
} // copyValidPtrByMask

struct Region
{  // Expect multiple fields, common def & stride
   FieldDef elemDef[3], blkDef0, grdDef0;
   size_t  n;
   ConstFieldPtr f0;

//public:
   bool validate (const BMFieldInfo *pI) // SubRegion/Box ???
   {
      int nD= 0;
      if (pI)
      {
         //if (pI->)
         copyValidPtrByMask(&f0, 1, pI->field, pI->fieldMask);
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
            n= prodNI(elemDef,3);
         }
      }
      return((nD > 0) && (NULL != f0.p));
   } // validate (as conditional 'CTOR')

   bool collapsable (void) const { return(0 == (elemDef[0] & VT_BLKM)); }

   dim3 blockDef (void) { return dim3(blkDef0, 1, 1); }
   dim3 gridDef (void) { return dim3(grdDef0, elemDef[1], elemDef[2]); }
   int collapsedBlockDef (void) { return(blkDef0); }
   int collapsedGridDef (void) { if (blkDef0 > 0) { return((n + blkDef0-1) / blkDef0); } else return(0); }
};


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
   CUDAFieldMap (const void * p, const BinMapF32 *pM) : CUDAMap<T_Elem>(pM) { pF= (T_Elem *)p; }
   CUDAFieldMap (const void * p, const BinMapF64 *pM) : CUDAMap<T_Elem>(pM) { pF= (T_Elem *)p; }

   //__device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::operator () (pF[i]); }
   __device__ uint operator () (const size_t i) const { return CUDAMap<T_Elem>::eval(pF[i]); }
}; // template class CUDAFieldMap

class CUDAOrg
{
   FieldStride fs[3];
   uint  rowElem;
   BMStride rowWS, planeWS;   // 32b word strides

   void setDS (const FieldStride *pS, const FieldDef *pD)
   {
      if (pS)
      {  // copy
         for (int i=0; i<3; i++) { fs[i]= pS[i]; }
      }
      else if (pD)
      {  // Generate stride
         FieldStride k= 1;
         for (int i=0; i<3; i++)
         {
            fs[i]= k;
            k*= pD[i];
         }
      }
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


template <typename T_Elem>
class CUDAMultiField
{  // Multiple fields with common stride
protected:
   const T_Elem   * fPtr[BMFI_FIELD_MAX];
   uint           nF;

   uint setF (const ConstFieldPtr a[], const uint mask)
   {
      return copyValidPtrByMask( (ConstFieldPtr*)fPtr, BMFI_FIELD_MAX, a, mask);
   } // setF

public:
   CUDAMultiField (const BMFieldInfo *pI)
   {
      if (pI->pD)
      {
         nF= setF(pI->field, pI->fieldMask);
      }
   } // CTOR

   //__device__ size_t index (uint x, uint y, uint z) const { return(x * stride[0] + y * stride[1] + z * stride[2]); }

   __device__ T_Elem sum (const size_t i) const
   {
      T_Elem s= (fPtr[0])[i];
      for (int iF=1; iF < nF; iF++) { s+= (fPtr[iF])[i]; }
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


#ifdef NO_WLP

// local mem bit merge util
__device__ int merge32 (BMPackWord w[32], const int j)
{
   if (0 == (j & 0x3))
   {  // j : { 0, 4, 8, 12, 16, 10, 24, 28 } 8P 3I
      for (int l=1; l<4; l++) { w[j]|= w[j+l]; }

      __syncthreads(); // Required for (unexplained) divergence

      if (0 == (j & 0xF))
      {  // j : { 0, 16 } 2P 3I
         for (int l=4; l<16; l+=4) { w[j]|= w[j+l]; }

         __syncthreads(); //  Optional ?
         if (0 == j) { w[0]|= w[16]; }
      }
   }
   return(j);
} // merge32

template <typename T_Elem>
__global__ void mapField (BMPackWord rBM[], const CUDAFieldMap<T_Elem> f, const size_t n)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   __shared__ BMPackWord w[VT_BLKN];
   if (i < n)
   {
      const uint j= threadIdx.x & VT_WRDM;   // lane index (bit number)
      const uint k= threadIdx.x & ~VT_WRDM;  // warp index (word number)

      w[threadIdx.x]= f(i) << j;

      if (0 == merge32(w+k, j)) { rBM[i>>VT_WRDS]= w[threadIdx.x]; }
   }
} // mapField

#endif


/***/

#ifdef WLP_BALLOT

template <typename T_Elem>
__global__ void mapFieldB (BMPackWord rBM[], const CUDAFieldMap<T_Elem> f, const size_t n)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n)
   {
      uint w= __ballot_sync(VT_WRDM, f(i) ); // Nice simplification but doesn't work... ?

      if (0 == (threadIdx.x & VT_WRDM)) { rBM[i>>VT_WRDS]= w; }
   }
} // mapFieldB

#endif // WLP_BALLOT

__device__ uint bitMergeWarp (uint w)
{  // CUDA9 warp level primitives (supported on CUDA7+ ? )
   for (int s= warpSize/2; s > 0; s>>= 1)
   {  // TODO - FIX - warning: integer conversion resulted in a change of sign
      w|= (uint)__shfl_down_sync(-1, w, s);
   }
   return(w);
} // bitMergeWarp

template <typename T_Elem>
__global__ void mapField (BMPackWord rBM[], const CUDAFieldMap<T_Elem> f, const size_t n)
{
   const size_t i= blockIdx.x * blockDim.x + threadIdx.x;
   if (i < n)
   {
      const uint j= threadIdx.x & VT_WRDM;   // lane index (bit number)

      uint w= bitMergeWarp( f(i) << j );

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

      uint w= bitMergeWarp( mf(i) << j );

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

      uint w= bitMergeWarp( f(i) << j );

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

      uint w= bitMergeWarp( mf(i) << j );

      if (0 == j) { rBM[org.bmIndex(x, blockIdx.y, blockIdx.z)]= w; }
   }
} // mapStrideMultiField


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
   Region reg;

   //if (32 != warpSize) { WARN("[binMapCUDA] warpSize=%d\n", warpSize); }
   if (reg.validate(pF))
   {
      CTimerCUDA t;
      const char * pID= NULL;

      if ( setBMO(pO, reg.elemDef, pF->profID) )
      {
         switch (pF->profID & 0x30)
         {
/*          {  const int nRows= prodNI(fd.def+1,2);
               for (int i=0; i<nRows; i++)
               {  // Horribly inefficient iteration
                  mapField<<<reg.nBlk,VT_BLKN>>>(pW + i * pO->rowWS, CUDAFieldMap<float>(fd.field[0].pF32 + i * fd.stride[1], pMC), fd.def[0]);
               }
               pID= "nRows*mapField()";
            }  break; */
            case 0x00 :
            if (reg.collapsable())
            {
               mapField<<< reg.collapsedGridDef(), reg.collapsedBlockDef() >>>(pW, CUDAFieldMap<float>(reg.f0.pF32, pMC), reg.n);
               pID= "mapField()";
               break;
            } // else...
            case 0x10 :
               mapStrideField<<< reg.gridDef(), reg.blockDef() >>>(pW, CUDAOrg(pO, pF), CUDAFieldMap<float>(reg.f0.pF32, pMC));
               pID= "mapStrideField()";
               break;

            case 0x20 :
            if (reg.collapsable())
            {
               mapMultiField<<< reg.collapsedGridDef(), reg.collapsedBlockDef() >>>(pW, CUDAMultiFieldMap<float>(pF, pMC), reg.n);
               pID= "mapMultiField()";
               break;
            } // else ...
            case 0x30 :
               mapStrideMultiField<<< reg.gridDef(), reg.blockDef() >>>(pW, CUDAOrg(pO, pF), CUDAMultiFieldMap<float>(pF, pMC));
               pID= "mapStrideMultiField()";
               break;
         }
         LOG("binMapCUDA() - %s<<<%u>>>() - dt= %Gms\n", pID, reg.collapsedBlockDef(), t.elapsedms());
         if (0 == ctuErr(NULL, pID)) { return(pO); }
      }
   }
   return(NULL);
} // binMapCUDA
