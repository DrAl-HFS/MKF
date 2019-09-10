// utilCUDA.h - CUDA API utils
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-June 2019

#ifndef CUDA_UTIL_HPP
#define CUDA_UTIL_HPP

//include "ctUtil.h"

#ifdef __NVCC__ // CUDA_API ???

#define CUDA_TIMER_EVENT_BITS 1
#define CUDA_TIMER_EVENTS    (1<<CUDA_TIMER_EVENT_BITS)
#define CUDA_EVENT_NUM_MASK  BIT_MASK(CUDA_TIMER_EVENT_BITS)

class CTimerCUDA
{
protected:
   cudaError_t r; // For debug
   cudaEvent_t e[CUDA_TIMER_EVENTS];
   int n;

public:
   CTimerCUDA (void)
   {
      n= 0;
      for (int i=0; i<CUDA_TIMER_EVENTS; i++) { r= cudaEventCreate(e+i); }
   } // CTor

   ~CTimerCUDA ()
   {
      for (int i=0; i<CUDA_TIMER_EVENTS; i++) { r= cudaEventDestroy(e[i]); }
   } // DTor

   void stampStream (void) { r= cudaEventRecord( e[ n++ & CUDA_EVENT_NUM_MASK ] ); }

   float elapsedms (bool stamp=true, bool sync=true)
   {  float ms=0;
      if (stamp) { stampStream(); }
      if (n>1)
      {
         int i0= (n-2) & CUDA_EVENT_NUM_MASK;
         int i1= (n-1) & CUDA_EVENT_NUM_MASK;
         if (sync) { r= cudaEventSynchronize( e[i1] ); }
         r= cudaEventElapsedTime(&ms, e[ i0 ], e[ i1 ]);
      }
      return(ms);
   } // elapsedms
}; // CTimerCUDA

#endif // __NVCC__

#endif // CUDA_UTIL_HPP
