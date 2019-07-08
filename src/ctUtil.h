// ctUtil.h - NB: .cu assumes c++ style compilation

#ifndef CT_UTIL_H
#define CT_UTIL_H

#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   uint nF, bytesF;
   long nU, bytesU;
   long nZ, bytesZ;
   float *pDF, *pHF;  // Device / Host ptrs
   uint  *pDU, *pHU;
   void *pDZ, *pHZ;
} Context;

extern int cuBuffAlloc (Context *pC, uint n);
extern int cuBuffRelease (Context *pC);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CT_UTIL_H
