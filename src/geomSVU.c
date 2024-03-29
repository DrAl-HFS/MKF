// geomSVU.h - assorted scalar vector related hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#include "geomSVU.h"


int *setKNI (int r[], const int n, const int k)
{
   for (int i=0; i<n; i++) { r[i]= k; }
   return(r);
} // setKNI

int *copyNI (int r[], const int n, const int v[])
{
   for (int i=0; i<n; i++) { r[i]= v[i]; }
   return(r);
} // copyNI

int *addKNI (int r[], const int a[], const int n, const int k)
{
   for (int i=0; i<n; i++) { r[i]= a[i]+k; }
   return(r);
} // addKNI

int sumNI (const int v[], int n)
{  //long
   int s= v[0];
   for (int i=1; i<n; i++) { s+= v[i]; }
   //if ((int)s != s) { ERROR_CALL("() - overflow (0x%x)\n", s); }
   return(s);
} // sumNI

int prodNI (const int v[], int n)
{  //long
   int x= v[0];
   for (int i=1; i<n; i++) { x*= v[i]; }
   //if ((int)x != x) { ERROR_CALL("() - overflow (0x%x)\n", x); }
   return(x);
} // prodNI

int mergeMinMaxNI (int rMin[], int rMax[], const int minA[], const int maxA[], const int minB[], const int maxB[], const int n)
{
   int d=0;
   for (int i=0; i<n; i++) { rMin[i]= MIN(minA[i], minB[i]); rMax[i]= MAX(maxA[i], maxB[i]); d+= rMax[i] > rMin[i]; }
   return(d);
} // mergeMinMaxNI


/***/

size_t sumNZ (const size_t a[], const size_t n)
{
   size_t s= a[0];   // assumes n>0 !
   for (size_t i=1; i<n; i++) { s+= a[i]; }
   return(s);
} // sumNZ


/***/

float *scaleFNI (float r[], const int n, const int v[], const float s)
{
   for (int i=0; i<n; i++) { r[i]= v[i] * s; }
   return(r);
} // scaleFNI


/***/

float *setKNF (float r[], const int n, const float k)
{
   for (int i=0; i<n; i++) { r[i]= k; }
   return(r);
} // setKNF

float *copyNF (float r[], const int n, const float v[])
{
   for (int i=0; i<n; i++) { r[i]= v[i]; }
   return(r);
} // cpyNF

float *absNF (float r[], const int n, const float v[])
{
   for (int i=0; i<n; i++) { r[i]= fabs(v[i]); }
   return(r);
} // absNF

float *sqrNF (float r[], const int n, const float v[])
{
   for (int i=0; i<n; i++) { r[i]= v[i] * v[i]; }
   return(r);
} // sqrNF

float sumNF (const float v[], int n)
{
   float s= v[0];
   for (int i=1; i<n; i++) { s+= v[i]; }
   return(s);
} // sumNF

float sqrMag3D (const float dx, const float dy, const float dz) { return(dx*dx + dy*dy + dz*dz); }

float sep3D (const float a[3], const float b[3])
{
   float s2= sqrMag3D( a[0]-b[0], a[1]-b[1], a[2]-b[2] );
   if (s2 > 0) { return sqrtf(s2); }
   //else
   return(0);
} // sep3D
