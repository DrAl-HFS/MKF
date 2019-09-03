// geomSVU.h - assorted scalar vector related hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#ifndef GEOM_SVU_H
#define GEOM_SVU_H

#include "report.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Integer */

//typedef float GeomScalar;
extern int *setKNI (int r[], const int n, const int k);

extern int *copyNI (int r[], const int n, const int v[]);

extern int *addKNI (int r[], const int a[], const int n, const int k);

// reducing
extern int sumNI (const int v[], int n);
extern int prodNI (const int v[], int n);

extern int mergeMinMaxNI (int rMin[], int rMax[], const int minA[], const int maxA[], const int minB[], const int maxB[], const int n);

/* Hybrid */

extern float *scaleFNI (float r[], const int n, const int v[], const float s);

/* Float */

//extern float *repNF (float f[], const int n, const float k);
extern float *setKNF (float r[], const int n, const float k);

extern float *copyNF (float r[], const int n, const float v[]);

extern float *absNF (float r[], const int n, const float v[]);

extern float *sqrNF (float r[], const int n, const float v[]);

// reducing

extern float sumNF (const float v[], int n);

extern float sqrMag3D (const float dx, const float dy, const float dz); // { return(dx*dx + dy*dy + dz*dz); }

extern float sep3D (const float a[3], const float b[3]);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEOM_SVU_H
