// binMapUtil.h - packed binary map general utility code.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef BIN_MAP_UTIL_H
#define BIN_MAP_UTIL_H

#include "report.h"

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


/***/

// Define map (comparison operator) using a string
extern void setBinMapF32 (BinMapF32 *pC, const char relopChars[], const float t);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // BIN_MAP_UTIL_H

