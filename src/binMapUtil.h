// binMapUtil.h - packed binary map general utility code.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef BIN_MAP_UTIL_H
#define BIN_MAP_UTIL_H

#include "geomSVU.h"
//#include "report.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of threshold values per mapping: 1 (simple threshold) or 2 (interval / hysteresis)
#define BM_NUMT   (1)
#define BM_FLAG_T2IVL  (1<<15)

// Comparison operation descriptor (single threshold).
// The possible 0 or 1 result of a comparison
// is obtained from m[] via a comparison-to-index
// function. Thus any relation < <= = != >= > can
// be described as a combination of three bits.
// The runtime cost is always two comparisons per
// threshold value plus the lookup.
typedef struct
{
   float t[BM_NUMT];
   int   m;   // 3 bits for simple threshold, 3*3= 9 for interval, plus flags.
   // NB: 32bit alignment (pre-NVIDIA-Pascal).
} BinMapF32;

#define BMC_GT (0x4)
#define BMC_EQ (0x2)
#define BMC_LT (0x1)
#define BMC_NV (0x0)
#define BMC_AL (0x7)

typedef struct
{
   uint row, plane;
} BMStrideDesc;

typedef long int Stride;
typedef struct
{
   Stride s[2];
} Stride2;

typedef struct
{
   Stride s[3];
} Stride3;

#define MFC_FIELD_MAX 4

typedef union { void *p; float *pF32; double *pF64; size_t w; } ScalarPtr;
typedef struct
{
   Stride3 stride;
    // Entire field, possibly interleaved
   ScalarPtr field[MFC_FIELD_MAX];
/*   union { // NB: always DEVICE mem ptrs!
      const void * p[MFC_FIELD_MAX];
      const float  * pF32[MFC_FIELD_MAX];
      const double * pF64[MFC_FIELD_MAX];
   };*/
} MultiFieldDesc;

typedef struct
{  struct { // anon structs - influence packing ?
      struct { uint8_t  nField, elemBits, opr, profile; };
      int       def[3]; }; // Def to process, possibly sub region of larger field
   MultiFieldDesc mfd;
} MultiFieldInfo;


/***/

// Define map (comparison operator) using a string
extern void setBinMapF32 (BinMapF32 *pC, const char relopChars[], const float t);

extern size_t setBMSD (BMStrideDesc *pSD, const int def[3], const char profID);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BIN_MAP_UTIL_H

