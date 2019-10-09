// geomHacks.c - basic 3D geometry hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-October 2019

#include "geomHacks.h"
#include "geomRaster.hpp"

/***/

float sphereArea (const float r) { return(4 * M_PI * r*r); }
float sphereVol (const float r) { return((4 * M_PI / 3) * r*r*r); }

float sphereCapArea (const float a, const float h) { return(M_PI * (a*a + h*h)); }
float sphereCapVol (const float a, const float h) { return((M_PI / 6) * h * (3*a*a + h*h)); }

float blockArea (const float r[3]) { return(8 * (r[0]*r[1] + r[1]*r[2] + r[0]*r[2])); }
float blockVol (const float r[3]) { return(8 * r[0]*r[1]*r[2]); }


int intersectSS1 (IntersectSS *pI, const float rA, const float rB, const float sAB)
{
   pI->dA= pI->a= 0;
   if (sAB <= rA + rB)
   {
      if (0 != sAB)
      {
         const float r2s= (0.5 / sAB);
         const float s2AB= sAB*sAB;
         const float r2A= rA*rA;
         const float t= r2A + s2AB - rB*rB;
         const float t2= t*t;
         const float d2= 4 * s2AB * r2A;
         pI->dA= r2s * t;
         if (d2 >= t2)
         {
            if (d2 == t2) { return(ID_TANGENT); }
            //else
            pI->a= r2s * sqrtf( d2 - t2 );
            return(ID_INTERSECT);
         }
      }
      return(ID_ENCLOSE);
   }
   return(ID_DISTINCT);
} // intersectSS1


typedef struct
{
   float r, c[3];
} Ball3D;


int measureScaledBB (float m[2], const Ball3D b[2], const float mScale)
{
   const float r0= b[0].r * mScale;
   const float r1= b[1].r * mScale;
   const float s= sep3D(b[0].c, b[1].c) * mScale;
   IntersectSS ss;
   const int t= intersectSS1(&ss, r0, r1, s);

   if (ID_ENCLOSE == t)
   {
      float r= MAX(r0, r1);
      m[GHM_V]= sphereVol(r);
      m[GHM_S]= sphereArea(r);
   }
   else
   {
      m[GHM_S]= sphereArea(r0) + sphereArea(r1);
      m[GHM_V]= sphereVol(r0) + sphereVol(r1);
      if (ID_INTERSECT == t)
      {
         float h0, h1;
         h0= r0 - ss.dA;
         h1= r1 - (s - ss.dA);
         m[GHM_V]-= sphereCapVol(ss.a, h0) + sphereCapVol(ss.a, h1);
         m[GHM_S]-= sphereCapArea(ss.a, h0) + sphereCapArea(ss.a, h1);
      }
   }
   return(t);
} // measureScaledBB

/***/

int rangeNI (int mm[], const int x[], const int n)
{  //if (n > 0)
   mm[0]= mm[1]= x[0];
   for (int i=1; i<n; i++)
   {
      mm[0]= MIN(mm[0], x[i]);
      mm[1]= MAX(mm[1], x[i]);
   }
   return(mm[1] - mm[0]);
} // rangeNI

int midRangeNI (const int x[], const int n)
{
   int mm[2];
   rangeNI(mm, x, n);
   return((mm[0] + mm[1]) >>1);
} // midRangeNI

float divF32 (float n, float d) { if (0 != d) return(n / d); else return(0); }
float sumRcpNI32 (const int d[], const int n)
{
   float s= divF32(1.0, d[0]);
   for (int i=1; i<n; i++) { s+= divF32(1.0, d[i]); }
   return(s);
} // sumRcpNI32

float midRangeHNI (const int x[], const int n)
{
   int mm[2];
   rangeNI(mm, x, n);
   return divF32(2, sumRcpNI32(mm, 2) );
} // midRangeNI

//I64 prodSumA1VN (const int v[], const int a, const int n)
I64 prodOffsetNI (const int x[], const int n, const int o)
{
   I64 r= x[0] + o;
   for (int i=1; i<n; i++) { r*= x[i] + o; }
   return(r);
} // prodSumA1VN


/***/


//int dummyBrk (void) { static int nd=0; return(++nd); }

size_t genPattern (GHMeasure *pM, void *pV, const int def[3], NumEnc enc, int nF, uint8_t id, const float param[3])
{
   const char *name[]={"empty","ball","solid","box","balls"};
   size_t n, nE= prodNI(def,3);
   const float size= param[0];
   float r[3], scale= 1.0 / midRangeNI(def,3);
   Ball3D b[2];
   GeomParam gp;
   RasParam rp={0,}; // {{0.0, 1.0}},RAS_FLAG_FLOAT|32};
   GHMeasure ghm={0,0};
   int t=0, bits=0;

   n= encSizeN(&bits, nE * MAX(1,nF), enc);
   rp.enc= enc;
   switch(enc)
   {
      case ENC_F32 :
      case ENC_F64 :
         rp.wF[0]= param[2]; rp.wF[1]= param[1];
         break;

      default : // WARN_CALL();
         rp.wI[0]= param[2]; rp.wI[1]= param[1];
         break;

      case ENC_U1 :
         rp.wI[0]= 0; rp.wI[1]= 1;
         rp.flags= RAS_FLAG_WRAL;
         break;
   }

   memset(&gp, 0, sizeof(gp));
   switch(id)
   {
      case 4 :
         b[0].r= 0.2 * size;
         b[1].r= 0.3 * size;
         for (int d=0; d<3; d++) { b[0].c[d]= 0.4 * def[d]; b[1].c[d]= 0.6 * def[d]; }
         t= measureScaledBB(ghm.m, b, scale);
         // set rasterisation param
         gp.id= 0x3; //GEOM_BALL
         gp.nObj= 2;
         gp.vF[gp.nF++]= b[0].r; // {r,c(x,y,z)}
         copyNF(gp.vF+gp.nF, 3, b[0].c); gp.nF+= 3;
         gp.vF[gp.nF++]= b[1].r; // {r,c(x,y,z)}
         copyNF(gp.vF+gp.nF, 3, b[1].c); gp.nF+= 3;
         break;
      case 3 :
         setKNF(r,3,0.5*size*scale);
         ghm.m[GHM_S]= blockArea(r);
         ghm.m[GHM_V]= blockVol(r);
         // set rasterisation param
         gp.id= 0x5; //GEOM_BOX
         gp.nObj= 1;
         setKNF(gp.vF+gp.nF, 3, 0.5*size); gp.nF+= 3; // {r(x,y,z),c(x,y,z)}
         scaleFNI(gp.vF+gp.nF, 3, def, 0.5); gp.nF+= 3;
         break;
      case 2 :
         ghm.m[GHM_V]= 1;
         memset(pV, -1, n);
         break;
      case 1 :
         b[0].r= 0.5 * size;
         ghm.m[GHM_S]= sphereArea(b[0].r*scale);
         ghm.m[GHM_V]= sphereVol(b[0].r*scale);
         // set rasterisation parameters
         gp.id= 0x3; //GEOM_BALL
         gp.nObj= 1;
         gp.vF[gp.nF++]= 0.5 * size; // {r,c(x,y,z)}
         scaleFNI(gp.vF+gp.nF, 3, def, 0.5); gp.nF+= 3;
         break;
      default :
         id= 0;
         memset(pV, 0, n);
         break;
   }
   //n= 0;
   //dummyBrk();
   //LOG("gp: id=%d, n=%d, nF=%d\n", gp.id, gp.nObj, gp.nF);
   if ((gp.id > 0) && (gp.nObj > 0))
   {
      if (0 == (rp.flags & RAS_FLAG_WRAL))
      {  // Lazy write needs clean buffer...
         memset(pV, rp.wI[0], n);
      }
      n= rasterise(pV, def, &gp, &rp);
   }
   LOG("def[%d,%d,%d] %s(%G)->%d,%zu (/%d=%G(PC), anyl.meas.: S%G V%G\n", def[0], def[1], def[2], name[id], size, t, n, nE, (F64)n / nE, ghm.m[GHM_S], ghm.m[GHM_V]);
   if (pM) { *pM= ghm; }
   return(n);
} // genPattern

void geomTest (float rA, float rB)
{
   GHMeasure ghm;
   Ball3D b[2]={0};
   int t;

geomObjTest();

   b[0].r= rA; b[1].r= rB;

   LOG("\n***\ngeomTest() / intersectSS1(%G, %G, ..)\n", rA, rB);
   LOG("Area: %G %G\n", sphereArea(rA), sphereArea(rB));
   for (float sAB= rA + rB + 0.5; sAB>=-0.5; sAB-= 0.5)
   {
      b[1].c[0]= sAB;
      t= measureScaledBB(ghm.m, b, 1);
      LOG("%G -> t=%d S%G V%G\n", sAB, t, ghm.m[GHM_S], ghm.m[GHM_V]);
   }
} // geomTest
