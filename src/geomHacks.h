// geomHacks.h - basic 3D geometry hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-October 2019

#ifndef GEOM_HACKS_H
#define GEOM_HACKS_H

#include "geomSVU.h"
#include "encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

// Try to keep this as basic as possible to limit feature/dependancy bloat.
// Complex stuff belongs in a proper geometry/math library...

/***/

// Basic formulae
extern float sphereArea (const float r); // 4.pi.r^2
extern float sphereVol (const float r);  // 4/3.pi.r^3

extern float sphereCapArea (const float a, const float h); // pi.(a^2+h^2)
extern float sphereCapVol (const float a, const float h); // pi/6.h.(3.a^2+h^2))

extern float blockArea (const float r[3]); // 4.(r0.r1+r1.r2+r0.r2)
extern float blockVol (const float r[3]);  // 8.r0.r1.r2


/***/

#define ID_DISTINCT  (0)
#define ID_TANGENT   (1)
#define ID_INTERSECT (2)
#define ID_ENCLOSE   (3)

typedef struct
{  // Circle of intersection -
   float dA, a; // - distance from centre A, radius
} IntersectSS;
// Intersection formulae - sphere-sphere collapsed 1D test
extern int intersectSS1 (IntersectSS *pI, const float rA, const float rB, const float sAB);


/***/

extern int rangeNI (int mm[], const int x[], const int n);

extern int midRangeNI (const int x[], const int n);
extern float midRangeHNI (const int x[], const int n);

// PRODUCT(v[i] + a) Used for image definition: (X-1)(Y-1)(Z-1) point elements -> cells
extern I64 prodOffsetNI (const int x[], const int n, const int o); //*pO);


/***/

#define GHM_S 0
#define GHM_V 1
typedef struct { float m[2]; } GHMeasure;

extern size_t genPattern (GHMeasure *pM, void *pV, const int def[3], NumEnc enc, int nF, uint8_t id, const float param[3]);

extern void geomTest (float, float);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEOM_HACKS_H
