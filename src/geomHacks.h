// geomHacks.h - basic 3D geometry hacks.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-July 2019

#ifndef GEOM_HACKS_H
#define GEOM_HACKS_H

#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

// Try to keep this as basic as possible to limit feature/dependancy bloat.
// Complex stuff belongs in a proper geometry/math library...

/***/

// Basic formulae
extern float sphereVol (const float r); // { return((4 * M_PI / 3) * r * r *r); }
extern float boxVol (const float r); // { return(8 * r * r * r); }

extern float mag2 (const float dx, const float dy, const float dz); // { return(dx*dx + dy*dy + dz*dz); }

/***/

// Rasterise primitive into scalar field
extern int genBall (float f[], const int def[3], const float r);

extern int genBlock (float f[], const int def[3], const float r);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEOM_HACKS_H
