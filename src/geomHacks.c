// geomHacks.c - basic 3D geometry hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-July 2019

#include "geomHacks.h"

/***/

float sphereVol (const float r) { return((4 * M_PI / 3) * r * r *r); }

float boxVol (const float r) { return(8 * r * r * r); }

float mag2 (const float dx, const float dy, const float dz) { return(dx*dx + dy*dy + dz*dz); }


/***/

int genBall (float f[], const int def[3], const float r)
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
            if (mag2(j-c[0], k-c[1], l-c[2]) <= r2) { f[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBall

int genBlock (float f[], const int def[3], const float r)
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
            if ( (abs(j-c[2]) <= r) && (abs(k-c[1]) <= r) && (abs(l-c[0]) <= r) ) { f[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBox
