// geomHacks.c - basic 3D geometry hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-July 2019

#include "geomHacks.h"

/***/

float sphereVol (const float r) { return((4 * M_PI / 3) * r * r *r); }

float boxVol (const float r) { return(8 * r * r * r); }

float mag2 (const float dx, const float dy, const float dz) { return(dx*dx + dy*dy + dz*dz); }


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
            if (mag2(j-c[0], k-c[1], l-c[2]) <= r2) { f[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBall

size_t genBlock (float f[], const int def[3], const float r)
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
} // genBlock

float genPattern (float f[], int id, const int def[3], const float param)
{
   const char *name[]={"empty","ball","solid","box"};
   size_t n, nF= def[0] * def[1] * def[2];
   float vr=0, fracR= param / def[1];

   n= nF;
   switch(id)
   {
      case 3 :
         vr= boxVol(fracR);
         n= genBlock(f, def, param);
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
