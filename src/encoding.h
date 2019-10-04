// encoding.h - Numeric encoding descriptor for scalar field flexibility
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Oct. 2019

#ifndef ENCODING_H
#define ENCODING_H

#include "util.h"

#ifdef __cplusplus
extern "C" {
#endif


#define ENC_MASK_TYP 0xF0
#define ENC_MASK_NUM 0x0F
#define ENC_TYPE_FPB 0xF0
#define ENC_TYPE_INT 0xE0
#define ENC_TYPE_BTS 0x00

// Common data types
#define ENC_F32 (ENC_TYPE_FPB|4)
#define ENC_F64 (ENC_TYPE_FPB|8)
#define ENC_U8  (ENC_TYPE_BTS|8)
#define ENC_U1  (ENC_TYPE_BTS|1)

// Aliases
#define ENC_FLOAT  ENC_F32
#define ENC_DOUBLE ENC_F64

typedef uint8_t NumEnc;


// Return storage size in bytes of n elements, optionally set bit size of single element
extern size_t encSizeN (int *pElemBits, const size_t n, const NumEnc e);

extern int encAlignment (const NumEnc e);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ENCODING_H
