// mkfAcc.c - tools for calculating Minkowski Functionals on a scalar field (via packed binary map and pattern distribution).
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-Sept 2019

#include "mkfACC.h"

#ifdef __PGI
#include <openacc.h> // -> /opt/pgi/linux86-64/2019/include/openacc.h
// PGI: _DEF_OPENACC
// GNU: _OPENACC_H
#define OPEN_ACC_API
#endif

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
   const BMPackWord * restrict pR0, // Location within row of first -
   const BMPackWord * restrict pR1, //  - and second planes
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
   const BMPackWord * restrict pR0, // Location within row of first -
   const BMPackWord * restrict pR1, //  - and second planes
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
   const BMPackWord * restrict pRow[2],
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

int setupAcc (int id)
{  // Only multicore acceleration works presently: GPU produces garbage...
   int t, r=-1;
#ifdef OPEN_ACC_API
   if (1 == id) { t= acc_device_nvidia; } else { t= acc_device_host; }
   acc_set_device_type( t );
   r= acc_get_device_type();
   LOG_CALL("(%d) - acc_*_device_type() - %d -> %d\n", id, t, r);
#endif
   return(r);
} // setupAcc

Bool32 mkfAccGetBPFDSimple
(
   size_t   rBPFD[MKF_BINS],
   BMPackWord * restrict pW,
   const MKFAccScalar * restrict pF,
   const FieldDef def[3],
   const MKFAccBinMap *pM
)
{
   const int rowStride= BITS_TO_WRDSH(def[0],CHUNK_SHIFT);
   const int planeStride= rowStride * def[1];
   //const int volStride= planeStride * def[2];
   const int nF= prodNI(def,3);

   #pragma acc data  present_or_create( pW[:(planeStride * def[2])] ) \
                     present_or_copyin( pF[:nF], def[:3], pM[:1] )  \
                     copy( rBPFD[:MKF_BINS] )
   {  // #pragma acc parallel vector ???
      if ((rowStride<<5) == def[0])
      {  // Multiple of 32
         binMapAcc(pW, pF, nF, pM);
      }
      else
      {
         binMapRowsAcc(pW, pF, def[0], rowStride, def[1]*def[2], pM);
      }

      for (int j= 0; j < (def[2]-1); j++)
      {
         const BMPackWord * restrict pPlane[2];
         pPlane[0]= pW + j * planeStride;
         pPlane[1]= pW + (j+1) * planeStride;
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
   return(TRUE);
} // mkfAccGetBPFDSimple

#ifdef MKF_ACC_CUDA_INTEROP

//include <openacc.h> // -> /opt/pgi/linux86-64/2019/include/openacc.h

#include "mkfCUDA.h"
//#include "binMapCUDA.h"

Bool32 mkfAccCUDAGetBPFD
(
   size_t   rBPFD[MKF_BINS],  // Result (Binary Pattern Frequency Distribution)
   const void        * pF,  // Scalar field (input) as opaque element type (see enc)
   const FieldDef    def[3],          // Definition (dimensions) of scalar field
   const NumEnc       enc,    // element type
   const MKFAccBinMap * pM
)
{
   void *p= NULL;

   acc_set_device_type( acc_device_nvidia ); // HACKY
   if (acc_device_nvidia == acc_get_device_type())
   {
      BMPackWord * pW;
      BMOrg bmo;
      const size_t nF= prodNI(def,3);
      const size_t nBM= setBMO(&bmo, def, 0); //(planeStride * def[2])
      BMFieldInfo fi={0};
      ConstFieldPtr table[1];

      // allocated via OpenACC for CUDA access to array/field data (others passed to
      // kernels using API "value parameter auto-marshalling" into const memory)
      //LOG("mkfAccCUDAGetBPFD() - sizeof(BMFieldInfo)=%d\n", sizeof(BMFieldInfo));
      fi.fieldTableMask=  0x01;
      fi.elemID= enc;
      //fi.opr=
      //fi.profID=
      fi.pD= def;
      //fi.pS= NULL; // auto stride
      fi.pFieldDevPtrTable= table;
      switch(enc)
      {
         case ENC_F32 :
         {
            const float * restrict pF32= (const float *)pF;
            #pragma acc data present_or_copyin( pF32[:nF] )
            {
               #pragma acc host_data use_device( pF32 ) // get OpenACC device memory pointers
               {
                  table[0].pF32= pF32;
                  if (pW= binMapCUDA(NULL, NULL, &bmo, &fi, pM))
                  {
                     p= mkfCUDAGetBPFDH(NULL, rBPFD, &bmo, pW, MKFCU_PROFILE_FAST);
                  }
               }
            }
            break;
         } // case ENC_F32

         case ENC_F64 :
         {
            const double * restrict pF64= (const double *)pF;
            #pragma acc data present_or_copyin( pF64[:nF] )
            {
               #pragma acc host_data use_device( pF64 ) // get OpenACC device memory pointers
               {
                  table[0].pF64= pF64;
                  if (pW= binMapCUDA(NULL, NULL, &bmo, &fi, pM))
                  {
                     p= mkfCUDAGetBPFDH(NULL, rBPFD, &bmo, pW, MKFCU_PROFILE_FAST);
                  }
               }
            }
            break;
         } // case ENC_F64

      } // switch
   }
   return(NULL != p);
} // mkfAccCUDAGetBPFD

#endif // MKF_ACC_CUDA_INTEROP
