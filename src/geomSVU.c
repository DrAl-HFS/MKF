// geomSVU.h - assorted scalar vector related hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#include "geomSVU.h"

float *repNF (float f[], const int n, const float k)
{
   for (int i=0; i<n; i++) { f[i]= k; }
   return(f);
} // repNF

float sqrMag3D (const float dx, const float dy, const float dz) { return(dx*dx + dy*dy + dz*dz); }

float sep3D (const float a[3], const float b[3])
{
   float s2= sqrMag3D( a[0]-b[0], a[1]-b[1], a[2]-b[2] );
   if (s2 > 0) { return sqrtf(s2); }
   //else
   return(0);
} // sep3D
