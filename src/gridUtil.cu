// gridUtil.cu - Factored out from binMap*
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Nov 2019

#include "gridUtil.cuh"


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

#define W32M 0x1F

bool GridRegion::validate (const BMFieldInfo *pI) // TODO: SubRegion/Box ???
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
         //blkDef0= VT_BLKN;
         //if (blkDim0...
         grdDef0= (elemDef[0] + (blkDef0-1) ) / blkDef0;
         nElem= prodNI(elemDef,3);
         if ( (elemDef[0] == nElem) ||
            ( (0 == (elemDef[0] & W32M)) &&
               ((NULL == pI->pS) || (REGION_NDIM == planarity(elemDef, pI->pS, REGION_NDIM)) ) ) )
         {
            flags|= REGION_F_AS1D;
         }
      }
      return((nDim > 0) && (nField > 0));
   }
   return(false);
} // validate (as conditional 'CTOR')

/*
GridPlane::GridPlane (const GridRegion& reg, const BMFieldInfo *pI)
{
   grid[0]= reg.grdDef0;
   grid[1]= reg.elemDef[1];
   nElem= prodNI(reg.elemDef, 2);
   copyOrGenStride(&stepF, 1, 2, pI->pD, pI->pS);
} // CTOR

GridSlab::GridSlab (const GridRegion& reg, const BMOrg *pO, const BMFieldInfo *pI) : GridPlane(reg, pI)
{
   stepWBM= pO->planeWS;
   if (reg.nSlab > 1) { depth= reg.elemDef[2] / reg.nSlab; }
   else { depth= reg.elemDef[2]; }
} // CTOR

*/
