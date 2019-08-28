#include "geomObj.hpp"


int IGeom3DObj::mergeBoundsI (int l[], int u[], int m, const GeomBoundsID b) const // override
{
   int lt[3], ut[3], t= boundsI(lt, ut, MIN(3,m));
   //if (t > 3) ERROR
   if (GB_OUT == b) { return mergeMinMaxNI(l, u, l, u, lt, ut, m); }
   else { return mergeMinMaxNI(u, l, u, l, ut, lt, m); }
   //return(-1);
} // IGeom3DObj::mergeBoundsI

CGeomAgg::CGeomAgg (int n)
{
   aIGOPtr= NULL;
   nIGO= 0;
   if (n > 0)
   {
      aIGOPtr= new IGeomObj*[n];
      if (aIGOPtr)
      {
         nIGO= n;
         for (int i=0; i<n; i++) { aIGOPtr[i]= NULL; }
      }
   }
} // CGeomAgg

CGeomAgg::~CGeomAgg ()
{
   for (int i=0; i<nIGO; i++)
   {
      if (aIGOPtr[i]) { delete aIGOPtr[i]; }
      //aIGOPtr[i]= NULL;
   }
   nIGO= 0;
} // CGeomAgg::~CGeomAgg

bool CGeomAgg::inI (const int vI[]) const // override
{
   for (int i=0; i<nIGO; i++)
   {  // conjunction assumed
      if (aIGOPtr[i])
      {
         if (aIGOPtr[i]->inI(vI)) { return(true); }
      }
   }
   return(false);
} // CGeomAgg::inI

bool CGeomAgg::inF (const float vF[]) const // override
{
   for (int i=0; i<nIGO; i++)
   {  // conjunction assumed
      if (aIGOPtr[i])
      {
         if (aIGOPtr[i]->inF(vF)) { return(true); }
      }
   }
   return(false);
} // CGeomAgg::inF

int CGeomAgg::boundsI (int l[GEOM_DMAX], int u[GEOM_DMAX], int m) const
{
   int t, a= 0, d= 0;
   for (int i=0; i<nIGO; i++)
   {  // union
      const IGeomObj *pG= aIGOPtr[i];
      if (pG)
      {
         if (0 == a) { t= pG->boundsI(l, u, m); } else { t= pG->mergeBoundsI(l, u, m, GB_OUT); }
         a+= (t >= 0);
         d= MAX(t, d);
      }
   }
   return(d);
} // CGeomAgg::boundsI

int CGeomAgg::mergeBoundsI (int l[], int u[], int m, const GeomBoundsID b) const // override
{
   int lt[GEOM_DMAX], ut[GEOM_DMAX], t= boundsI(lt, ut, MIN(GEOM_DMAX, m));
   if (GB_OUT == b) { return mergeMinMaxNI(l, u, l, u, lt, ut, m); }
   else { return mergeMinMaxNI(u, l, u, l, ut, lt, m); }
} // CGeomAgg::mergeBoundsI

/***/

IGeomObj *CGeomFactory::create1 (GeomID id, const float fParam[], const int nFP)
{
   IGeomObj *pI= NULL;
   int nR= MIN(1,nFP);
   switch(id)
   {
      case GEOM_SPHERE :
         pI= new CGeomSphere(GeomSV3(fParam, nR, 0), GeomSV3(fParam+nR, nFP-nR, 0));
         break;
      case GEOM_SPHERE_SHELL :
         nR= MIN(2,nFP);
         pI= new CGeomSphere(GeomSV3(fParam, nR, 0), GeomSV3(fParam+nR, nFP-nR, 0));
         break;
      case GEOM_BALL :
         pI= new CGeomSphere(GeomSV3(fParam, nR, 0), GeomSV3(fParam+nR, nFP-nR, 0));
         break;
      case GEOM_CUBE :
         pI= new CGeomBox(GeomSV3(fParam, nR, 1), GeomSV3(fParam+nR, nFP-nR, 0));
         break;
      case GEOM_BOX :
         nR= MIN(3,nFP);
         pI= new CGeomBox(GeomSV3(fParam, nR, 1), GeomSV3(fParam+nR, nFP-nR, 0));
         break;
   }
   return(pI);
} // CGeomFactory::create1

IGeomObj *CGeomFactory::createN (GeomID id, const int nObj, const float fParam[], const int nFP)
{
   if ((nObj > 0) && (nFP > nObj))
   {
      if (1 == nObj) { return create1(id,fParam, nFP); }
      //else
      CGeomAgg *pA= new CGeomAgg(nObj);
      if (pA)
      {
         int iFP= 0;
         int nPO= nFP / nObj;
         for (int i=0; i<nObj; i++)
         {
            pA->aIGOPtr[i]= create1(id, fParam+iFP, nPO);
            iFP+= nPO;
         }
         return(pA);
      }
   }
   return(NULL);
} // CGeomFactory::createN

IGeomObj *CGeomFactory::release (IGeomObj *p)
{
   if (p) { delete p; p= NULL; }
   return(p);
} // CGeomFactory::release
