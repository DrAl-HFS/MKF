// gridUtil.hpp - Factored out from binMap*
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Nov 2019

#include "binMapUtil.h"

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
   GridRegion (int blkD) { blkDef0= blkD; }
   bool validate (const BMFieldInfo *pI); // TODO: SubRegion/Box ???

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
