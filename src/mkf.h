// mkf.h - NB: .cu assumes c++ style compilation

#ifndef MKF_H
#define MKF_H

#include "ctUtil.h"

#ifdef __cplusplus
extern "C" {
#endif


#define MC_GT (1<<2)
#define MC_EQ (1<<1)
#define MC_LT (1<<0)

typedef struct
{
   float t[1];
   int   m;
} MKBMapF32;

typedef size_t MKCount;

//MKCount rBPD[256]
//extern "C" int mkfProcess (const Context *pC, const int def[3], const MKBMapF32 *pMC);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_H
