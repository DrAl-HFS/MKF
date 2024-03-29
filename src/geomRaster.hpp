// geomRaster.hpp - rasterisation of basic geometric objects
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#ifndef GEOM_RASTER_HPP
#define GEOM_RASTER_HPP

#include "encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GEOM_PARAM_MAXV (8)

typedef struct
{
   int   id;
   int   nObj;
   float vF[GEOM_PARAM_MAXV];
   int   nF;
} GeomParam;

#define RAS_FLAG_WRAL   (0x80)

typedef struct
{
   union { float wF[2]; int wI[2]; }; // indices from geometry membership test (bool) 0=out, 1=in
   NumEnc  enc;
   uint8_t flags, pad[2];
} RasParam;

//extern "C"
extern size_t rasterise (void *pB, const int def[3], const GeomParam *pGP, const RasParam *pRP);

//extern "C"
extern void geomObjTest (void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // GEOM_RASTER_HPP
