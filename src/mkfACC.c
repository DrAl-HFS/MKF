// mkfAcc.c - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "mkfACC.h"

#ifndef ACC_INLINE
#define ACC_INLINE
#endif

// Processing chunks of 32 elements (i.e. bits packed into word)
// allows GPU hardware to be more easily/efficiently exploited.
#define CHUNK_SHIFT (5)
#define CHUNK_SIZE (1<<CHUNK_SHIFT)
#define CHUNK_MASK (CHUNK_SIZE-1)


/***/

// Load chunk from 4 adjacent rows within 2 neighbouring planes of binary map
ACC_INLINE void loadChunkSh0
(
   U64 bufChunk[4],     // chunk buffer
   const U32 * restrict pR0, // Location within row of first -
   const U32 * restrict pR1, //  - and second planes
   const U32 rowStride       // stride between successive rows within each plane
)
{  // vect
   bufChunk[0]= pR0[0];
   bufChunk[1]= pR0[rowStride];
   bufChunk[2]= pR1[0];
   bufChunk[3]= pR1[rowStride];
} // loadChunk

// As above but appending to last bits of preceding chunk
ACC_INLINE void loadChunkSh1
(
   U64 bufChunk[4],     // chunk buffer
   const U32 * restrict pR0, // Location within row of first -
   const U32 * restrict pR1, //  - and second planes
   const U32 rowStride       // stride between successive rows within each plane
)
{  // vect
   bufChunk[0] |= ( (U64) pR0[0] ) << 1;
   bufChunk[1] |= ( (U64) pR0[rowStride] ) << 1;
   bufChunk[2] |= ( (U64) pR1[0] ) << 1;
   bufChunk[3] |= ( (U64) pR1[rowStride] ) << 1;
} // loadChunk

#if 0
// store binary patterns to temp
ACC_INLINE int buildPattern (U8 bufPatt[], U64 bufChunk[4], const int n)
{
   for (int i= 0; i < n; i++) // seq
   {
      U8 r[4]; // temporary to permit/encourage vectorisation

      for (int k= 0; k < 4; k++) // vect
      {
         r[k]= (bufChunk[k] & 0x3) << (2*k);
         bufChunk[k] >>= 1;
      }
      bufPatt[i]= r[0] | r[1] | r[2] | r[3];
   }
   return(n);
} // buildPattern
#endif

// Update frequency distribution of binary patterns
// Transpose pattern assembly : better parallelisability?
ACC_INLINE void addPattern (size_t rBPFD[256], U64 bufChunk[4], const int n)
{
   for (int i= 0; i < n; i++) // seq
   {
      U8 r[4]; // temporary to permit/encourage vectorisation

      for (int k= 0; k < 4; k++) // vect
      {
         r[k]= (bufChunk[k] & 0x3) << (2*k);
         bufChunk[k] >>= 1;
      }
      rBPFD[(r[0] | r[1] | r[2] | r[3])]++;
   }
} // addPattern

// "Conventional" pattern assembly in 4bit slices
ACC_INLINE void addPatternOM (size_t rBPFD[256], U64 bufChunk[4], const int n)
{
   U8 r=0;

   r= (bufChunk[0] & 0x1) |
      ((bufChunk[1] & 0x1) << 1) |
      ((bufChunk[2] & 0x1) << 2) |
      ((bufChunk[3] & 0x1) << 3);
   for (int i= 0; i < n; i++) // seq
   {
      for (int k= 0; k < 4; k++) // vect
      {
         bufChunk[k] >>= 1;
         r|= (bufChunk[k] & 0x1) << (4+k);
      }
      rBPFD[r]++; r>>= 4;
   }
} // addPatternOM

#define ADD_PATTERN addPattern
//OM

// Add a single row of 8bit/3D patterns (defined by four adjacent rows of
// elements) to the result distribution array. Efficient parallel execution
// seems unlikely due to memory access patterns:-
//  i) requires *atomic_add()* to distribution array
// ii) successive chunks need merge of leading/trailing bits
ACC_INLINE void addRowBPFD
(
   size_t rBPFD[256], // result pattern distribution
   const U32 * restrict pRow[2],
   const int rowStride,
   const int n    // Number of single bit elements packed in row
)
{  // seq
   int m, k, i; // , j
   U64 bufChunk[4]= { 0,0,0,0 };
   //U8 bufPatt[CHUNK_SIZE];

   // First chunk of n bits yields n-1 patterns
   loadChunkSh0(bufChunk, pRow[0]+0, pRow[1]+0, rowStride);
   ADD_PATTERN(rBPFD, bufChunk, MIN(CHUNK_SIZE-1, n-1));
   // Subsequent whole chunks yield n patterns
   i= 0;
   m= n>>CHUNK_SHIFT;
   while (++i < m)
   {
      loadChunkSh1(bufChunk, pRow[0]+i, pRow[1]+i, rowStride);
      ADD_PATTERN(rBPFD, bufChunk, CHUNK_SIZE);
   }
   // Check for residual bits < CHUNK_SIZE
   k= n & CHUNK_MASK;
   if (k > 0)
   {
      loadChunkSh1(bufChunk, pRow[0]+i, pRow[1]+i, rowStride);
      ADD_PATTERN(rBPFD, bufChunk, k);
   }
} // addRowBPFD


/***/

int mkfAccGetBPFDSimple
(
   size_t   rBPFD[MKF_BINS],
   U32          * restrict pBM,
   const F32   * restrict pF,
   const int   def[3],
   const BinMapF32   * const pC
)
{
   const int rowStride= BITS_TO_WRDSH(def[0],CHUNK_SHIFT);
   const int planeStride= rowStride * def[1];
   //const int volStride= planeStride * def[2];
   const int nF= def[0]*def[1]*def[2];

   #pragma acc data  present_or_create( pBM[:(planeStride * def[2])] ) \
                     present_or_copyin( pF[:nF], def[:3], pC[:1] )  \
                     copy( rBPFD[:MKF_BINS] )
   {  // #pragma acc parallel vector ???
      if ((rowStride<<5) == def[0])
      {  // Multiple of 32
         binMapNF32(pBM, pF, nF, pC);
      }
      else
      {
         binMapRowsF32(pBM, pF, def[0], rowStride, def[1]*def[2], pC);
      }

      for (int j= 0; j < (def[2]-1); j++)
      {
         const U32 * restrict pPlane[2];
         pPlane[0]= pBM + j * planeStride;
         pPlane[1]= pBM + (j+1) * planeStride;
         #pragma acc loop seq
         for (int i= 0; i < (def[1]-1); i++)
         {
            const U32 * restrict pRow[2];
            pRow[0]= pPlane[0] + i * rowStride;
            pRow[1]= pPlane[1] + i * rowStride;
            addRowBPFD(rBPFD, pRow, rowStride, def[0]);
         }
      }
   }
   return(1);
} // mkfAccGetBPFDSimple

#ifdef MKF_ACC_CUDA_INTEROP

#include <openacc.h> // -> /opt/pgi/linux86-64/2019/include/openacc.h

#include "mkfCUDA.h"
//#include "binMapCUDA.h"

int mkfAccCUDAGetBPFD (size_t rBPFD[MKF_BINS], U32 * pBM, const F32 * pF, const int def[3], const BinMapF32 * const pMC)
{
   const int nF= def[0]*def[1]*def[2];
   BMStrideDesc sd[1];
   const size_t nBM= setBMSD(sd, def, 0);
   //(planeStride * def[2])
   acc_set_device_num( 0, acc_device_nvidia ); // HACKY
   #pragma acc data present_or_create( pBM[:nBM] ) present_or_copyin( pF[:nF], def[:3], sd[:1], pMC[:1] ) copy( rBPFD[:MKF_BINS] )
   {
      #pragma acc host_data use_device(pBM, pF, def, sd, pMC)
      binMapCudaRowsF32(pBM, pF, def[0], sd[0].row, def[1] * def[2], pMC);
      #pragma acc host_data use_device(rBPFD, def, sd, pBM)
      mkfCUDAGetBPFD(rBPFD, def, sd, pBM);
   }
   return(1);
} // mkfAccCUDAGetBPFD

#endif // MKF_ACC_CUDA_INTEROP
