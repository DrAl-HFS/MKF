// geomObj.hpp - abstraction of basic geometric object definitions
// and a very limited selection of properties. Remember: KISS.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#ifndef GEOM_OBJ_HPP
#define GEOM_OBJ_HPP

#include "geomSVU.h"

// 3D vector wrapper for more readable constructor assigments
struct GeomSV3
{
   //union { struct { float x, y, z; }; float v[3]; }; // Feasible, but not yet useful and possibly undesireable?
   float v[3];

   GeomSV3 (const float a=0) { v[0]= v[1]= v[2]= a; }
   GeomSV3 (const float a[], const int n, const float k=0) { copyNF(v, MIN(3,n), a); setKNF(v+n, 3-MAX(0,n), k); }
   GeomSV3 (const float x, const float y, const float z) { v[0]= x; v[1]= y; v[2]= z; }

   void assign (float a[3], int n=3) const { copyNF(a, n, v); }
   void assignAbs (float a[3], int n=3) const { absNF(a, n, v); }
   void assignSqr (float a[3], int n=3) const { sqrNF(a, n, v); }
}; // GeomSV3


/***/

// Interface (abstract base) is dimensionally agnostic...
struct IGeomObj
{
   virtual bool inI (const int i[]) const = 0; // { return inF(i[0], i[1], i[2]); }
   virtual bool inF (const float f[]) const = 0; // { return inF(f[0], f[1], f[2]); }
   virtual int boundsI (int l[], int u[]) const = 0;
   virtual int safeBoundsI (int l[], int u[]) const = 0;
}; // IGeomObj

// Implementations have implicit dimensionality...

class CGeomSphere : public IGeomObj
{
protected :
   float r2[2], c[3];

public :
   CGeomSphere (const GeomSV3& ar=GeomSV3(1,0,0), const GeomSV3& lc=GeomSV3()) { ar.assignSqr(r2,2); lc.assign(c); }

   bool inI (const int i[]) const override { return inF3(i[0], i[1], i[2]); }
   bool inF (const float f[]) const override { return inF3(f[0], f[1], f[2]); }

   int boundsI (int l[], int u[]) const override
   {
      float r= sqrt(r2[0]); // + eps ?
      int d=0;
      for (int i=0; i<3; i++) { l[i]= c[i]-r; u[i]= c[i]+r; d+= (u[i] > l[i]); }
      return(d);
   } // boundsI

   int safeBoundsI (int l[], int u[]) const override
   {
      int lt[3], ut[3];
      boundsI(lt,ut);
      return mergeMinMaxNI(u, l, u, l, ut, lt, 3);
   } // safeBoundsI

   bool inF3 (float x, float y, float z) const
   {
      const float s2= sqrMag3D(x-c[0], y-c[1], z-c[2]) ;
      return((r2[0] >= s2) && (r2[1] <= s2));
   }
}; // CGeomSphere


class CGeomBox : public IGeomObj
{
protected :
   float r[3], c[3];

public :
   CGeomBox (const GeomSV3& ar=GeomSV3(1), const GeomSV3& lc=GeomSV3()) { ar.assignAbs(r); lc.assign(c); }

   bool inI (const int i[]) const override { return inF3(i[0], i[1], i[2]); }
   bool inF (const float f[]) const override { return inF3(f[0], f[1], f[2]); }

   int boundsI (int l[], int u[]) const override
   {
      int d=0;
      for (int i=0; i<3; i++) { l[i]= c[i]-r[i]; u[i]= c[i]+r[i]; d+= (u[i] > l[i]); }
      return(d);
   } // boundsI

   int safeBoundsI (int l[], int u[]) const override
   {
      int lt[3], ut[3];
      boundsI(lt,ut);
      return mergeMinMaxNI(u, l, u, l, ut, lt, 3);
   } // safeBoundsI

   bool inF3 (float x, float y, float z) const
   {
      return( (fabs(x-c[0]) <= r[0]) && (fabs(y-c[1]) <= r[1]) && (fabs(z-c[2]) <= r[2]) );
   }
}; // CGeomBox

/***/

// Unscoped enum equivalent to macro definition
enum GeomID { GEOM_NONE, GEOM_SPHERE, GEOM_BOX };

class CGeomFactory
{
protected :

public:
   CGeomFactory () { ; }
   ~CGeomFactory () { ; }

   IGeomObj *create (GeomID id, const float param[], const int nParam)
   {
      IGeomObj *pI= NULL;

      if (nParam > 0)
      {
         int nR;

         switch(id)
         {
            case GEOM_SPHERE :
               nR= nParam - 3;
               nR= MAX(0, nR);
               pI= new CGeomSphere(GeomSV3(param, nR, 0), GeomSV3(param+1, nParam-1, 0));
               break;
            case GEOM_BOX :
               nR= nParam - 3;
               nR= MAX(0, nR);
               pI= new CGeomBox(GeomSV3(param, nR, 1), GeomSV3(param+nR, nParam-nR, 0));
               break;
         }
      }
      return(pI);
   } // create

   IGeomObj *release (IGeomObj *p)
   {
      if (p) { delete p; p= NULL; }
      return(p);
   } // release

}; // CGeomFactory

#endif // GEOM_OBJ_HPP
