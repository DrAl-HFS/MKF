// binMapUtil.h - packed binary map general utility code.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#ifndef BIN_MAP_UTIL_H
#define BIN_MAP_UTIL_H

#include "geomSVU.h"
#include "encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

// Number of threshold values per mapping: 1 (simple threshold) or 2 (interval / hysteresis)
#define BM_NUMT   (1)
#define BM_TLB   (0)
#define BM_TUB   (0) // BM_TLB+1
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
   double t[BM_NUMT];
   uint   m;   // 3 bits for simple threshold, 3*3= 9 for interval, plus flags.
} BinMapF64;

// DEPRECATE?
typedef struct
{
   float t[BM_NUMT];
   uint  m;   // 3 bits for simple threshold, 3*3= 9 for interval, plus flags.
   // NB: 32bit alignment (pre-NVIDIA-Pascal).
} BinMapF32;


#define BMC_GT (0x4)
#define BMC_EQ (0x2)
#define BMC_LT (0x1)
#define BMC_NV (0x0)
#define BMC_AL (0x7)

typedef uint BMStride;
typedef uint BMPackWord;

typedef struct
{  // Memory organisation of BM
   uint  rowElem;
   uint  rowPairs, planePairs; // paired (N-1) counts
   BMStride rowWS, planeWS;   // 32b word strides
} BMOrg;

// To handle various buffer arrangements (parallel arrays vs. element/line/plane interleaving)
// declared in C or whatever, "type punned" pointers are used. This requires deep understanding
// of the machine representation of arrays, and as such is likely traumatic for non-CompSci folks.
// Helper functions are available below for this reason (but perhaps need better organisation).

// Could add "const half *pF16;" if relevant headers included, but perhaps not useful?
typedef union { const void *p; const float *pF32; const double *pF64; size_t w; } ConstFieldPtr;

typedef long int FieldStride; // 64bit just in case
typedef int    FieldDef;       // 32bit most compatible?
typedef struct
{
   struct { // anon struct used to influence packing
      uint32_t  fieldTableMask; // Bits 0..31 enable/disable entries in field device pointer table
      NumEnc   elemID;
      uint8_t  oprID, profID, flags;
   };
   const FieldDef     *pD;
   const FieldStride *pS;  // NULL => assume fully planar fields
   ConstFieldPtr      *pFieldDevPtrTable;
} BMFieldInfo;


/***/

// Define map (comparison operator) using a string
BinMapF64 *setBinMapF64 (BinMapF64 *pBM64, const char relopChars[], const float t);
//extern BinMapF32 *setBinMapF32 (BinMapF32 *pBM32, const char relopChars[], const float t);

size_t setBMO (BMOrg *pO, const FieldDef def[3], const char profID);

int copyNonZeroDef (FieldDef d[], const FieldDef *pD, const int nD);

int copyValidPtrByMask (ConstFieldPtr r[], const int max, const ConstFieldPtr a[], const uint mask);

int countValidPtrByMask (const ConstFieldPtr a[], uint mask);

// Generate n stride values from definition and base stride, skipping some if required.
// Returns number generated (zero for bad inputs)
int genStride (FieldStride fs[], const int n, const int start, const FieldDef *pD, FieldStride stride);

int copyOrGenStride (FieldStride fs[], const int n, const int start, const FieldDef *pD, const FieldStride *pS);

// HACKY REFACTORING #include "ctUtil.h"

#define BMCU_FIELD_MAX 4   // Number of fields actually supported

typedef struct
{  // Hacky perf/debug info
   float dtms[2];
} KernInfo;

typedef struct
{
   int nField, nElem;
   struct { NumEnc enc; uint8_t bytesElem, pad[2]; };
   long bytesF;
   long nU, bytesU;
   long nZ, bytesZ;
   void *pDF, *pHF;  // Device / Host ptrs
   BMPackWord  *pDU, *pHU;
   void *pDZ, *pHZ;
   BMOrg bmo;
   KernInfo ki;
} Context;

ConstFieldPtr *asFieldTab (const void **pp, const NumEnc id);

const BMFieldInfo *setupFields (BMFieldInfo *pI, void **ppF, const int nF, const int def[3], const int elemBytes, const int profile);

void **autoFieldPtr (ConstFieldPtr ptr[BMCU_FIELD_MAX], const Context *pC);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // BIN_MAP_UTIL_H

