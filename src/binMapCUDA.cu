// binMapCUDA.h - packed binary map generation from scalar fields
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Oct 2019

#include "binMapCUDA.h"
#include "utilCUDA.hpp"

// Templated device code factored out to header for reuse...
#include "binMapCUDA.cuh"


/* Misc functions */

int planarity (const FieldDef def[], const FieldStride stride[], int n)
{
   int i=0, r=0;
   if (n > 0)
   {
      FieldStride s= 1;
      do
      {
         r+= (s == stride[i]);
         s*= def[i];
      } while (++i < n);
   }
   return(r);
} // planarity


/* Host side utility class/struct */

#define REGION_NDIM   3
#define REGION_F_AS1D (0x01) // collapsable

struct GridRegion
{  // Expect multiple fields, common def & stride
   size_t   nElem;
   FieldDef  elemDef[REGION_NDIM];
   int32_t  grdDef0, blkDef0;
   uint8_t  nDim, nField, flags, nSlab; //, nStream;

//public:
   bool validate (const BMFieldInfo *pI) // TODO: SubRegion/Box ???
   {
      if (pI)
      {
         nDim= 0; nField= 0;
         nSlab= 0; // ??
         if (pI->pFieldDevPtrTable)
         {
            nField= countValidPtrByMask(pI->pFieldDevPtrTable, pI->fieldTableMask);
         }
         if (nDim= copyNonZeroDef(elemDef, pI->pD, REGION_NDIM))
         {
            blkDef0= VT_BLKN;
            //if (blkDim0...
            grdDef0= (elemDef[0] + (blkDef0-1) ) / blkDef0;
            nElem= prodNI(elemDef,3);
            if ( (elemDef[0] == nElem) ||
               ( (0 == (elemDef[0] & VT_WRDM)) &&
                  ((NULL == pI->pS) || (REGION_NDIM == planarity(elemDef, pI->pS, REGION_NDIM)) ) ) )
            {
               flags|= REGION_F_AS1D;
            }
         }
         return((nDim > 0) && (nField > 0));
      }
      return(false);
   } // validate (as conditional 'CTOR')

   bool collapsable (void) const { return(flags & REGION_F_AS1D); }

   dim3 blkDef (void) const { return dim3(blkDef0, 1, 1); }
   dim3 grdDef (void) const { return dim3(grdDef0, elemDef[1], elemDef[2]); }
   // 3D vs. 1D-collapsed launch params
   int blkDefColl (void) const { return(blkDef0); }
   int grdDefColl (void) const { if (blkDef0 > 0) { return((nElem + blkDef0-1) / blkDef0); } else return(0); }
   // Methods for hacky test code
   int blkDefRow (void) const { return(blkDef0); }
   int grdDefRow (void) const { return(grdDef0); }
#ifdef NO_WLP
   size_t blkShMem (void) { return( blkDef0 * sizeof(uint) ); }
#else
   size_t blkShMem (void) { return(0); }
#endif
}; // struct GridRegion

class GridPlane
{
protected:
   size_t     nElem;
   FieldDef    grid[2];
   FieldStride stepF;

   GridPlane (const GridRegion& reg, const BMFieldInfo *pI)
   {
      grid[0]= reg.grdDef0;
      grid[1]= reg.elemDef[1];
      nElem= prodNI(reg.elemDef, 2);
      copyOrGenStride(&stepF, 1, 2, pI->pD, pI->pS);
   } // CTOR
}; // class GridPlane

class GridSlab : private GridPlane
{
//protected:
public:
   BMStride stepWBM;
   FieldDef depth;

public:
   GridSlab (const GridRegion& reg, const BMOrg *pO, const BMFieldInfo *pI) : GridPlane(reg, pI)
   {
      stepWBM= pO->planeWS;
      if (reg.nSlab > 1) { depth= reg.elemDef[2] / reg.nSlab; }
      else { depth= reg.elemDef[2]; }
   } // CTOR

   FieldDef getResid (const GridRegion& reg) const
   {
      FieldDef r= reg.elemDef[2] - (reg.nSlab * depth);
      return MAX(0, r);
   } // setResid

   bool setDepth (FieldDef d) { if (d > 0) { depth= d; } return(d > 0); }

   size_t nElem (void) const { return(depth * GridPlane::nElem); }
   size_t stepW (void) const { return(depth * stepWBM); }
   size_t stepF (void) const { return(depth * GridPlane::stepF); }

   dim3 def (void) const { return dim3(GridPlane::grid[0], GridPlane::grid[1], depth); }
   dim3 defColl (void) const { return(GridPlane::grid[0] * GridPlane::grid[1] * depth); }
}; // class GridSlab


/* INTERFACE */
static BMPackWord * gpDevW= NULL;

extern "C"
BMPackWord *binMapCUDA
(
   KernInfo * pK,
   BMPackWord  * pW,
   BMOrg       * pO,
   const BMFieldInfo * pI,
   const BinMapF64   * pM
)
{
   GridRegion reg;

   //if (32 != warpSize) { WARN("[binMapCUDA] warpSize=%d\n", warpSize); }
   if (reg.validate(pI))
   {
      CTimerCUDA t;
      const char * pID= NULL;

      if ( setBMO(pO, reg.elemDef, pI->profID) )
      {
         if (NULL == pW)
         {
            if (NULL == gpDevW)
            {
               cudaMalloc(&(gpDevW), sizeof(BMPackWord) * pO->planeWS * (pO->planePairs+1));
               if (NULL == gpDevW) { return(NULL); } // else...
            }
            pW= gpDevW;
         }

         if ((0 == pI->profID) && reg.collapsable() && (1 == reg.nField))
         {
            switch (pI->elemID)
            {
               case ENC_F32 :
                  mapField<<< reg.grdDefColl(), reg.blkDefColl() >>>(pW, CUDAFieldMap<float>(pI, pM), reg.nElem);
                  pID= "mapField<F32>()";
                  break;
               case ENC_F64 :
                  mapField<<< reg.grdDefColl(), reg.blkDefColl() >>>(pW, CUDAFieldMap<double>(pI, pM), reg.nElem);
                  pID= "mapField<F64>()";
                  break;
            }
         }
         else
         switch (pI->profID & 0x70)
         {
            default : WARN("binMapCUDA() - profID=0x%02X, defaulting...\n", pI->profID);
            case 0x00 :
            if (reg.collapsable() && (1 == reg.nField))
            {
               //reg.nSlab= 3;
               if (reg.nSlab > 0)
               {
                  CUDAStrmBlk s;
                  CUDAFieldMap<float> map(pI, pM);
                  GridSlab slab(reg, pO, pI);
                  size_t wOffset= 0;
                  for (int i=0; i<reg.nSlab; i++)
                  {  //LOG("%d %d 0x%x %d %zu\n", i, slab.depth, wOffset, map.iOff, slab.nElem());
                     mapField<<< slab.defColl(), reg.blkDefColl(), 0, s[i] >>>(pW+wOffset, map, slab.nElem());
                     wOffset+= slab.stepW();
                     map.addOffset(slab.stepF());
                  }
                  if (slab.setDepth(slab.getResid(reg)))
                  {  //LOG("*R: %d 0x%x %d %zu\n", slab.depth, wOffset, map.iOff, slab.nElem());
                     mapField<<< slab.defColl(), reg.blkDefColl(), 0, s[0] >>>(pW+wOffset, map, slab.nElem());
                  }
                  pID= "nSlab*mapField()";
               }
               else
               {
                  mapField<<< reg.grdDefColl(), reg.blkDefColl() >>>(pW, CUDAFieldMap<float>(pI, pM), reg.nElem);
                  pID= "mapField()";
               }
               break;
            } // else...
            case 0x10 :
            if (1 == reg.nField)
            {
               mapStrideField<<< reg.grdDef(), reg.blkDef() >>>(pW, BinMapCUDAOrg(pO, pI), CUDAFieldMap<float>(pI, pM));
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
               mapStrideMultiField<<< reg.grdDef(), reg.blkDef() >>>(pW, BinMapCUDAOrg(pO, pI), CUDAMultiFieldMap<float>(pI, pM));
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
         if (pK) { pK->dtms[0]= t.elapsedms(); }
         else { LOG("binMapCUDA() - %s<<<%u>>>() - dt= %Gms\n", pID, reg.blkDefColl(), t.elapsedms()); }
         if (0 == ctuErr(NULL, pID)) { return(pW); }
      }
   }
   return(NULL);
} // binMapCUDA

extern "C"
void binMapCUDACleanup (void)
{
   if (gpDevW) { cudaFree(gpDevW); gpDevW= NULL; }
} // binMapCUDACleanup
