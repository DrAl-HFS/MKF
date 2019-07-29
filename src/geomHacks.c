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

float sqrMag3D (const float dx, const float dy, const float dz) { return(dx*dx + dy*dy + dz*dz); }
float sep3D (const float a[3], const float b[3])
{
   float s2= sqrMag3D(a[0]-b[0],a[1]-b[1],a[2]-b[2]);
   if (s2 > 0) { return sqrtf(s2); }
   //else
   return(0);
} // sep3D

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
   float v, a;
} VA3D;

int measureScaled (VA3D m[1], const Ball3D b[2], const float mScale)
{
   const float r0= b[0].r * mScale;
   const float r1= b[1].r * mScale;
   const float s= sep3D(b[0].c, b[1].c) * mScale;
   IntersectSS ss;
   const int t= intersectSS1(&ss, r0, r1, s);

   if (ID_ENCLOSE == t)
   {
      float r= MAX(r0, r1);
      m[0].v= sphereVol(r);
      m[0].a= sphereArea(r);
   }
   else
   {
      m[0].a= sphereArea(r0) + sphereArea(r1);
      m[0].v= sphereVol(r0) + sphereVol(r1);
      if (ID_INTERSECT == t)
      {
         float h0, h1;
         h0= r0 - ss.dA;
         h1= r1 - (s - ss.dA);
         m[0].v-= sphereCapVol(ss.a, h0) + sphereCapVol(ss.a, h1);
         m[0].a-= sphereCapArea(ss.a, h0) + sphereCapArea(ss.a, h1);
      }
   }
   return(t);
} // measureScaled

void testHack (float rA, float rB)
{
   Ball3D b[2]={0};
   VA3D m;
   int t;

   b[0].r= rA; b[1].r= rB;

   LOG("testHack() / intersectSS1(%G, %G, ..)\n", rA, rB);
   LOG("Area: %G %G\n", sphereArea(rA), sphereArea(rB));
   for (float sAB= rA + rB + 0.5; sAB>=-0.5; sAB-= 0.5)
   {
      b[1].c[0]= sAB;
      t= measureScaled(&m, b, 1);
      LOG("%G -> t=%d A=%G V=%G\n", sAB, t, m.a, m.v);
   }
} // testHack

/***/

I64 prodSumA1VN (const int v[], const int a, const int n)
{
   I64 r= v[0] + a;
   for (int i=1; i<n; i++) { r*= v[i] + a; }
   return(r);
} // prodSumA1VN

float *repNF (float f[], const int n, const float k)
{
   for (int i=0; i<n; i++) { f[i]= k; }
   return(f);
} // repNF


/***/

Bool32 inBall (const float x[3], const Ball3D *pB)
{
   return( sqrMag3D(x[0]-pB->c[0], x[1]-pB->c[1], x[2]-pB->c[2]) <= (pB->r * pB->r) );
} // inBall

size_t genNBall (float f[], const int def[3], const Ball3D *pB, const int nB)
{
   size_t i= 0, n= 0;
   int c[3];

   for (c[0]= 0; c[0]<def[0]; c[0]++)
   {
      for (c[1]= 0; c[1]<def[1]; c[1]++)
      {
         for (c[2]= 0; c[2]<def[2]; c[2]++)
         {
            float d[3]={c[0],c[1],c[2]};
            for (int b= 0; b < nB; b++)
            {
               if (inBall(d,pB+b))
               {
                  f[i]= 1.0; ++n;
                  break;
               }
            }
            ++i;
         }
      }
   }
   return(n);
} // genNBall

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

float genPattern (float f[], int id, const int def[3], const float param)
{
   const char *name[]={"empty","ball","solid","box","balls"};
   size_t n, nF= def[0] * def[1] * def[2];
   float r[3], scale= 1.0 / def[1];
   Ball3D b[2];
   VA3D m={0,0};
   int t=0;

   testHack(2,2);

   n= nF;
   switch(id)
   {
      case 4 :
         b[0].r= 0.55 * param;
         b[1].r= 0.45 * param;
         for (int d=0; d<3; d++) { b[0].c[d]= 0.45 * def[d]; b[1].c[d]= 0.55 * def[d]; }
         t= measureScaled(&m, b, scale);
         n= genNBall(f, def, b, 2);
         break;
      case 3 :
         repNF(r,3,param*scale);
         m.a= blockArea(r);
         m.v= blockVol(r);
         n= genBlock(repNF(r,3,param), def, r);
         break;
      case 2 :
         m.v= 1;
         memset(f, -1, sizeof(f[0])*nF);
         break;
      case 1 :
         m.a= sphereArea(param*scale);
         m.v= sphereVol(param*scale);
         b[0].r= param;
         for (int d=0; d<3; d++) { b[0].c[d]= 0.5 * def[d]; }
         n= genNBall(f, def, b, 1);
         break;
      default :
         id= 0;
         memset(f, 0, sizeof(f[0])*nF);
         break;
   }
   LOG("def[%d,%d,%d] %s(%G)->%d,%zu (/%d=%G, ref=%G)\n", def[0], def[1], def[2], name[id], param, t, n, nF, (F64)n / nF, m.v);
   return(m.v);
} // genPattern
