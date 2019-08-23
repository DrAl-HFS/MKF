// geomSVU.h - assorted scalar vector related hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#ifndef GEOM_SVU_H
#define GEOM_SVU_H

#include "report.h"

#ifdef __cplusplus
extern "C" {
#endif

//typedef float GeomScalar;

extern float *repNF (float f[], const int n, const float k);

extern float sqrMag3D (const float dx, const float dy, const float dz); // { return(dx*dx + dy*dy + dz*dz); }

extern float sep3D (const float a[3], const float b[3]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEOM_SVU_H
