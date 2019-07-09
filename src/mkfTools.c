// mkfTools.c - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "mkfTools.h"

#ifndef ACC_INLINE
#define ACC_INLINE
#endif

// Processing chunks of 32 elements (i.e. bits packed into word)
// allows GPU hardware to be more easily/efficiently exploited.
#define CHUNK_SHIFT (5)
#define CHUNK_SIZE (1<<CHUNK_SHIFT)
#define CHUNK_MASK (CHUNK_SIZE-1)


/***/

#include "weighting.inc"

/***/

// Load chunk from 4 adjacent rows within 2 neighbouring planes of binary map,
// appending to remaining bits of last chunk if required
ACC_INLINE void loadChunk
(
   U64 bufChunk[4],     // chunk buffer
   const U32 * restrict pR0, // Location within row of first -
   const U32 * restrict pR1, //  - and second planes
   const U32 rowStride,       // stride between successive rows within each plane
   const U32 lsh              // shift at which to append chunks (0/1)
)
{  // vect
   bufChunk[0] |= (pR0[0] << lsh);
   bufChunk[1] |= (pR0[rowStride] << lsh);
   bufChunk[2] |= (pR1[0] << lsh);
   bufChunk[3] |= (pR1[rowStride] << lsh);
} // loadChunk

#if 0
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
#else
ACC_INLINE void addPattern (U32 rBPD[256], U64 bufChunk[4], const int n)
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
      rBPD[r]++; r>>= 4;
   }
} // addPattern
#endif

// Add a single row of 8bit/3D patterns (defined by four adjacent rows of
// elements) to the result distribution array. Efficient parallel execution
// seems unlikely due to memory access patterns:-
//  i) requires *atomic_add()* to distribution array
// ii) successive chunks need merge of leading/trailing bits
ACC_INLINE void addRowBPD
(
   U32 rBPD[256], // result pattern distribution
   const U32 * restrict pRow[2],
   const int rowStride,
   const int n    // Number of single bit elements packed in row
)
{  // seq
   int m, k, i; // , j
   U64 bufChunk[4]= { 0,0,0,0 };
   //U8 bufPatt[CHUNK_SIZE];

   // First chunk of n bits yields n-1 patterns
   loadChunk(bufChunk, pRow[0]+0, pRow[1]+0, rowStride, 0);
   addPattern(rBPD, bufChunk, MIN(CHUNK_SIZE-1, n-1));
   //k= buildPattern(bufPatt, bufChunk, MIN(CHUNK_SIZE-1, n-1));
   //for (j=0; j<k; j++) { rBPD[ bufPatt[j] ]++; }
   // Subsequent whole chunks yield n patterns
   i= 0;
   m= n>>CHUNK_SHIFT;
   while (++i < m)
   {
      loadChunk(bufChunk, pRow[0]+i, pRow[1]+i, rowStride, 1);
      addPattern(rBPD, bufChunk, CHUNK_SIZE);
      //k= buildPattern(bufPatt, bufChunk, CHUNK_SIZE);
      //for (int j=0; j<k; j++) { rBPD[ bufPatt[j] ]++; }
   }
   // Check for residual bits < CHUNK_SIZE
   k= n & CHUNK_MASK;
   if (k > 0)
   {
      loadChunk(bufChunk, pRow[0]+i, pRow[1]+i, rowStride, 1);
      addPattern(rBPD, bufChunk, k);
      //k= buildPattern(bufPatt, bufChunk, k);
      //for (int j=0; j<k; j++) { rBPD[ bufPatt[j] ]++; }
   }
} // addRowBPD

/***/

void procSimple (U32 rBPD[256], U32 * restrict pBM, const F32 * restrict pF, const int def[3], const BinMapF32 * const pC)
{
   const int rowStride= BITS_TO_WRDSH(def[0],CHUNK_SHIFT);
   const int planeStride= rowStride * def[1];
   //const int volStride= planeStride * def[2];
   const int nF= def[0]*def[1]*def[2];

   #pragma acc data present_or_create( pBM[:(planeStride * def[2])] ) present_or_copyin( pF[:nF], def[:3], pC[:1] ) copy( rBPD[:256] )
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
            addRowBPD(rBPD, pRow, rowStride, def[0]);
         }
      }
   }
} // procSimple


/***/

MKMeasureVal volFrac (const U32 h[256])
{
   size_t s[2]={0,0};
   for (int i= 0; i<256; i+= 2)
   {
      s[0]+= h[i];
      s[1]+= h[i+1];
   }
   LOG_CALL("() - s[]={%zu, %zu} (%zu)\n", s[0], s[1], s[0]+s[1]);
   return( s[1] / (MKMeasureVal)(s[0] + s[1]) );
} // volFrac

MKMeasureVal chiEP3 (const U32 h[256])
{
   I32 k=0;
   for (int i= 0; i<256; i++) { k+= (I32)gWEP3[i] * (I32)h[i]; }
   //LOG_CALL("() - k[]={%i, %i}\n", k[0], k[1]);
   return( (MKMeasureVal) k * M_PI / 6 );
} // chiEP3


