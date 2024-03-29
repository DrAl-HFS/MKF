// mkfAcc.h - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#ifndef MKF_ACC_H
#define MKF_ACC_H

#include "binMapACC.h"
#include "mkfUtil.h"

#ifdef __cplusplus
extern "C" {
#endif


/***/

int setupAcc (int id, int flags); // 1=NVIDIA, 0=multicore. flags&1= verbose

// Original version for reference / testing, can be used for host side computation* but features
// are presently limited:
//    no polymorphism,
//    no multi-field,
//    no sub-buffering of intermediate data (simple non-overlapped processing),
//    no strided access (i.e. planar only, no region-of-interest or borders).
// *NB - OpenAcc GPU acceleration DOES NOT WORK on this code for reasons yet to be determined.
Bool32 mkfAccGetBPFDSimple
(
   size_t   rBPFD[MKF_BINS],   // Result (Binary Pattern Frequency Distribution)
   BMPackWord * restrict pW,  // Intermediate data (packed bitmap - result of binarising scalar field)
   const MKFAccScalar * restrict pF, // Scalar field (input)
   const FieldDef def[3],             // Definition (dimensions) of scalar field
   const MKFAccBinMap * pM // Mapping (binarisation)
);

#ifdef MKF_ACC_CUDA_INTEROP

// Launch CUDA kernels to process OpenACC buffers - more features avail...
Bool32 mkfAccCUDAGetBPFD
(
   size_t     rBPFD[MKF_BINS],
   const void        * pF, // NB - opaque ("punned") element type!
   const FieldDef    def[3],
   const NumEnc       enc,
   const MKFAccBinMap * const pM
);

#endif // MKF_ACC_CUDA_INTEROP

#ifdef __cplusplus
} // extern "C"
#endif

#endif // MKF_ACC_H
