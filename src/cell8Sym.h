// cell8Sym.h - symmetry of 8 vertex (cubic) cell.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#ifndef CELL8SYM_H
#define CELL8SYM_H

#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
   uint8_t bits, count, nF, nE, nV; // pad[3];
} GroupInf;

extern int c8sGetPattern (uint8_t patBuf[256], GroupInf inf[32]);

extern int c8sGetMap (uint8_t map[256]);

extern void c8sTest (void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // CELL8SYM_H
