// mkfAcc.h - dumping ground for junk experiments.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#ifndef MKF_HACKS_H
#define MKF_HACKS_H

#include "cell8Sym.h"

#ifdef __cplusplus
extern "C" {
#endif


/***/

extern double sumProdZxF64 (const size_t z[], const double w[], const int n);

extern void symMS (float m[2], const size_t rBPFD[CELL8_SYMM_GROUPS]);

extern void measurePatternTest (int m);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_HACKS_H
