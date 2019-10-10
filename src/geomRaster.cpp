// geomRaster.hpp - rasterisation of basic geometric objects
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#include "geomObj.hpp"
#include "packedBFA.hpp"
#include "geomRaster.hpp"

size_t dot3I (const int i[3], const int j[3]) { return( (long int)i[0]*j[0] + i[1]*j[1] + i[2]*j[2] ); }

struct Trav3Inf
{
   int stride[3];
   int lb[3], ub[3];

   Trav3Inf (const int def[3], const int elemStride=1)
   {
      stride[0]= elemStride;
      stride[1]= stride[0] * def[0];
      stride[2]= stride[1] * def[1];
      setKNI(lb, 3, 0);
      addKNI(ub, def, 3, -1);
   }
   size_t index (const int idx[3]) const { return dot3I(idx, stride); }
}; // struct Trav3Inf

//void operator () (size_t idx, const float fVal) { pF32[idx]= fVal; }
template <typename T_Elem>
size_t traverseF (T_Elem f[], const IGeomObj *pG, const Trav3Inf& t, const T_Elem wF[2], const bool wral)
{
   size_t n=0;
   int i[3];

   for (i[2]= t.lb[2]; i[2] <= t.ub[2]; i[2]++)
   {
      for (i[1]= t.lb[1]; i[1] <= t.ub[1]; i[1]++)
      {
         for (i[0]= t.lb[0]; i[0] <= t.ub[0]; i[0]++)
         {
            bool k= pG->inI(i); n+= k;
            if (wral|k) { f[t.index(i)]= wF[k]; }
         }
      }
   }
   return(n);
} // traverseF

size_t traverseI (IWriteI& w, const IGeomObj *pG, const Trav3Inf& t, const int wI[2], const bool wral)
{
   size_t n=0;
   int i[3];

   for (i[2]= t.lb[2]; i[2] <= t.ub[2]; i[2]++)
   {
      for (i[1]= t.lb[1]; i[1] <= t.ub[1]; i[1]++)
      {
         for (i[0]= t.lb[0]; i[0] <= t.ub[0]; i[0]++)
         {
            bool k= pG->inI(i); n+= k;
            if (wral|k) { w.write(t.index(i), wI[k]); }
         }
      }
   }
   return(n);
} // traverseI


/* C INTERFACE */

extern "C" size_t rasterise (void *pV, const int def[3], const GeomParam *pGP, const RasParam *pRP)
{
   size_t n= 0;
   int bits;

   encSizeN(&bits, 1, pRP->enc);
   if (bits > 0)
   {
      CGeomFactory fG;
      IGeomObj *pG= fG.createN(GeomID(pGP->id & 0x7), pGP->nObj, pGP->vF, pGP->nF);
      if (pG)
      {
         Trav3Inf t(def);
         if (0 == pRP->flags & RAS_FLAG_WRAL) { pG->safeBoundsI(t.lb, t.ub, 3); LOG("safeBoundsI: l=%d %d %d u=%d %d %d", t.lb[0],t.lb[1],t.lb[2], t.ub[0],t.ub[1],t.ub[2]); }
         switch (pRP->enc)
         {
            case ENC_F32 :
               n= traverseF((float*)pV, pG, t, pRP->wF, pRP->flags & RAS_FLAG_WRAL);
               break;
            case ENC_F64 :
            {
               const double wF[2]= { pRP->wF[0], pRP->wF[1] };
               n= traverseF((double*)pV, pG, t, wF, pRP->flags & RAS_FLAG_WRAL);
               break;
            }
            default :
               // if (bits > 32) { WARN_CALL(); } else
               CWriteRL32P2B wr(pV, bits);
               n= traverseI(wr, pG, t, pRP->wI, pRP->flags & RAS_FLAG_WRAL);
               break;
         }
         pG= fG.release(pG); // no auto
      }
   }
   return(n);
} // rasterise

extern "C" void geomObjTest (void)
{
   uint8_t buf8[16];
   uint32_t buf32[2]={0,};
   CReadWriteRL32P2B rw(buf32,4);
   //IWriteI *pW= CWriteRL32P2B::rw;
   CGeomSphere s;
   IGeomObj *pO= &s;

   report(LOG0, "***\n%s() -\n", __func__);
   for (float x= -1.5; x <= 1.5; x+= 0.5)
   {
      GeomSV3 t(x,0,0);
      int r= pO->inF(t.v);
      report(LOG0, "s.inF(%G) - %d\n", x, r);
   }
   for (int i=0; i<=0xF; i++) { rw.write(i,i); }
   report(LOG0, "buf32: %X %X\nbuf8: ", buf32[0],buf32[1]);
   for (int i=0; i<=0xF; i++) { buf8[i]= rw.read(i); }
   reportBytes(LOG0, buf8, sizeof(buf8));
   report(LOG0, "\n***\n");
} // geomObjTest
