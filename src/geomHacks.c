// geomHacks.c - basic 3D geometry hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-July 2019

#include "geomHacks.h"

/***/

float sphereArea (const float r) { return(4 * M_PI * r*r); }
float sphereVol (const float r) { return((4 * M_PI / 3) * r*r*r); }

float sphereCapArea (const float a, const float h) { return(M_PI * (a*a + h*h)); }
float sphereCapVol (const float a, const float h) { return((M_PI / 6) * h * (3*a*a + h*h)); }

float blockArea (const float r[3]) { return(4 * (r[0]*r[1] + r[1]*r[2] + r[0]*r[2])); }
float blockVol (const float r[3]) { return(8 * r[0]*r[1]*r[2]); }

float sqrMag3 (const float dx, const float dy, const float dz) { return(dx*dx + dy*dy + dz*dz); }

int intersectSS1 (IntersectSS *pI, const float rA, const float rB, const float sAB)
{
   pI->dA= pI->a= 0;
   if (sAB <= rA + rB)
   {
      if (0 != sAB)
      {
         const float r2s= (0.5 / sAB);
         const float s2= sAB*sAB;
         const float rA2= rA*rA;
         const float t= rA2 + s2 - rB*rB;
         const float t2= t*t;
         const float d2= 4 * s2 * rA2;
         pI->dA= r2s * t;
         if (d2 >= t2)
         {
            if (d2 == t2) { return(ID_TANGENT); }
            //else
            pI->a= r2s * sqrt( d2 - t2 );
            return(ID_INTERSECT);
         }
      }
      return(ID_ENCLOSE);
   }
   return(ID_DISTINCT);
} // intersectSS1

/***/

I64 prodSumA1VN (const int v[], const int a, const int n)
{
   I64 r= v[0] + a;
   for (int i=1; i<n; i++) { r*= v[i] + a; }
   return(r);
} // prodSumA1VN


/***/

size_t genBall (float f[], const int def[3], const float r)
{
   size_t i= 0, n= 0;
   float c[3], r2= r * r;

   for (int d=0; d<3; d++) { c[d]= 0.5 * def[d]; }

   for (int j= 0; j<def[2]; j++)
   {
      for (int k= 0; k<def[1]; k++)
      {
         for (int l= 0; l<def[0]; l++)
         {
            if (sqrMag3(j-c[0], k-c[1], l-c[2]) <= r2) { f[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBall

size_t genBlock (float f[], const int def[3], const float r[3])
{
   size_t i= 0, n= 0;
   float c[3];

   for (int d=0; d<3; d++) { c[d]= 0.5 * def[d]; }

   for (int j= 0; j<def[2]; j++)
   {
      for (int k= 0; k<def[1]; k++)
      {
         for (int l= 0; l<def[0]; l++)
         {
            if ( (abs(j-c[2]) <= r[0]) && (abs(k-c[1]) <= r[1]) && (abs(l-c[0]) <= r[2]) ) { f[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBlock

void testHack (float rA, float rB)
{
   LOG("testHack() / intersectSS1() %s", "\n");
   for (float sAB= rA + rB + 0.5; sAB>=-0.5; sAB-= 0.5)
   {
      IntersectSS ss;
      float hA, hB;
      intersectSS1(&ss,rA,rB,sAB);
      hA= rA - ss.dA;
      hB= rB - (sAB - ss.dA);
      LOG("%G -> D: %G%+G rI: %G  A: %G%+G\n", sAB, ss.dA, sAB - ss.dA, ss.a, sphereCapArea(ss.a, hA), sphereCapArea(ss.a, hB));
   }
} // testHack

float genPattern (float f[], int id, const int def[3], const float param)
{
   const char *name[]={"empty","ball","solid","box"};
   size_t n, nF= def[0] * def[1] * def[2];
   float vr=0, fracR= param / def[1];

   testHack(2,4);

   n= nF;
   switch(id)
   {
      case 3 :
      {  float r[3]={fracR,fracR,fracR};
         vr= blockVol(r);
      }
      {  float r[3]={param,param,param};
         n= genBlock(f, def, r);
      }
         break;
      case 2 : vr= 1;
         memset(f, -1, sizeof(f[0])*nF);
         break;
      case 1 :
         vr= sphereVol(fracR);
         n= genBall(f, def, param);
         break;
      default :
         id= 0; vr= 0;
         memset(f, 0, sizeof(f[0])*nF);
         break;
   }
   LOG("def[%d,%d,%d] %s(%G)->%zu (/%d=%G, ref=%G)\n", def[0], def[1], def[2], name[id], param, n, nF, (F64)n / nF, vr);
   return(vr);
} // genPattern
