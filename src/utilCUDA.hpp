// utilCUDA.hpp - CUDA API utils
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Sept 2019

#ifndef CUDA_UTIL_HPP
#define CUDA_UTIL_HPP

//include "ctUtil.h"

#ifdef __NVCC__ // CUDA_API ???

#define CDTM_EVN_BITS   1
#define CDTM_EVN_COUNT  (1<<CDTM_EVN_BITS)
#define CDTM_EVN_MASK   BIT_MASK(CDTM_EVN_BITS)

//enum CDTMAction { CDTM_NONE, CDTM_STAMP, CDTM_STAMP_SYNC, CDTM_SYNC };
// Action Flags
#define CDTM_AF_STAMP  (0x01)
#define CDTM_AF_SYNC   (0x02)

class CTimerCUDA
{
protected:
   cudaError_t r; // For debug
   cudaEvent_t e[CDTM_EVN_COUNT];
   int n;

public:
   CTimerCUDA (int af=CDTM_AF_STAMP)
   {
      n= 0;
      for (int i=0; i < CDTM_EVN_COUNT; i++) { r= cudaEventCreate(e+i); }
      if (CDTM_AF_STAMP & af) { stampStream(); }
   } // CTor

   ~CTimerCUDA ()
   {
      for (int i=0; i < CDTM_EVN_COUNT; i++) { r= cudaEventDestroy(e[i]); }
   } // DTor

   void stampStream (void) { r= cudaEventRecord( e[ n++ & CDTM_EVN_MASK ] ); }
   //void syncStream (void) { r= cudaEventSynchronize( e[ n & CDTM_EVN_MASK ] ); }

   float elapsedms (int af=CDTM_AF_STAMP|CDTM_AF_SYNC)
   {  float ms=0;
      if (CDTM_AF_STAMP & af) { stampStream(); }
      if (n>1)
      {
         int i0= (n-2) & CDTM_EVN_MASK;
         int i1= (n-1) & CDTM_EVN_MASK;
         if (CDTM_AF_SYNC & af) { r= cudaEventSynchronize( e[i1] ); }
         r= cudaEventElapsedTime(&ms, e[ i0 ], e[ i1 ]);
      }
      return(ms);
   } // elapsedms
}; // CTimerCUDA

#endif // __NVCC__

#endif // CUDA_UTIL_HPP
