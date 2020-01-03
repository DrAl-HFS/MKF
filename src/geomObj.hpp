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

enum GeomBoundsID { GB_OUT, GB_IN };

// Interface (abstract base) is dimensionally agnostic...
struct IGeomObj
{
   virtual bool inI (const int vI[]) const = 0;
   virtual bool inF (const float vF[]) const = 0;
   virtual int boundsI (int l[], int u[], int m) const = 0;
   virtual int mergeBoundsI (int l[], int u[], int m, const GeomBoundsID b) const = 0;
   virtual int safeBoundsI (int l[], int u[], int m) const = 0;
}; // IGeomObj

// Explicit dimensional derivation allows some factoring of common features
struct IGeom3DObj : public IGeomObj
{
   bool inI (const int vI[3]) const override { return inF3(vI[0], vI[1], vI[2]); }
   bool inF (const float vF[3]) const override { return inF3(vF[0], vF[1], vF[2]); }
   //virtual int boundsI (int l[], int u[]) const = 0;
   int mergeBoundsI (int l[], int u[], int m, const GeomBoundsID b) const override;
   int safeBoundsI (int l[], int u[], int m) const override { return mergeBoundsI(l, u, m, GB_IN); }
   virtual bool inF3 (float, float, float) const = 0;
}; // IGeom3DObj


class CGeomSphere : public IGeom3DObj
{
protected :
   float r2[2], c[3];

public :
   CGeomSphere (const GeomSV3& ar=GeomSV3(1,0,0), const GeomSV3& lc=GeomSV3()) { ar.assignSqr(r2,2); lc.assign(c); }
   //   printf("CGeomSphere: R2=%G,%G, C=%G,%G,%G\n", r2[0], r2[1], c[0], c[1], c[2]); }

   int boundsI (int l[3], int u[3], int m) const override
   {
      float r= sqrt(r2[0]); // + eps ?
      int d=0;
      m= MIN(3,m);
      for (int i=0; i<m; i++) { l[i]= c[i]-r; u[i]= c[i]+r; d+= (u[i] > l[i]); }
      return(d);
   } // boundsI

   bool inF3 (float x, float y, float z) const override
   {
      const float s2= sqrMag3D(x-c[0], y-c[1], z-c[2]) ;
      return((r2[0] >= s2) && (r2[1] <= s2));
   }
}; // CGeomSphere


class CGeomBox : public IGeom3DObj
{
protected :
   float r[3], c[3];

public :
   CGeomBox (const GeomSV3& ar=GeomSV3(1), const GeomSV3& lc=GeomSV3()) { ar.assignAbs(r); lc.assign(c); }

   int boundsI (int l[3], int u[3], int m) const override
   {
      int d=0;
      m= MIN(3,m);
      for (int i=0; i<m; i++) { l[i]= c[i]-r[i]; u[i]= c[i]+r[i]; d+= (u[i] > l[i]); }
      return(d);
   } // boundsI

   bool inF3 (float x, float y, float z) const override
   {
      return( (fabs(x-c[0]) <= r[0]) && (fabs(y-c[1]) <= r[1]) && (fabs(z-c[2]) <= r[2]) );
   }
}; // CGeomBox

/***/

#define GEOM_DMAX (3)

class CGeomAgg : public IGeomObj
{
   friend class CGeomFactory;
protected :
   IGeomObj **aIGOPtr;
   int nIGO;

public :
   CGeomAgg (int n=0);
   ~CGeomAgg ();

   bool inI (const int vI[]) const override;

   bool inF (const float vF[]) const override;

   int boundsI (int l[], int u[], int m) const override;

   int mergeBoundsI (int l[], int u[], int m, const GeomBoundsID b) const override;

   int safeBoundsI (int l[], int u[], int m) const override { return mergeBoundsI(l, u, m, GB_IN); }

}; // CGeomAgg


/***/

// Unscoped enum equivalent to macro definition
enum GeomID { GEOM_NONE, GEOM_SPHERE, GEOM_SPHERE_SHELL, GEOM_BALL, GEOM_CUBE, GEOM_BOX };

class CGeomFactory
{
protected :

public :
   CGeomFactory () { ; }
   ~CGeomFactory () { ; }


   IGeomObj *create1 (GeomID id, const float fParam[], const int nFP);
   IGeomObj *createN (GeomID id, const int nObj, const float fParam[], const int nFP);
   IGeomObj *release (IGeomObj *p);

}; // CGeomFactory

#endif // GEOM_OBJ_HPP
