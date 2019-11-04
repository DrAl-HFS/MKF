// mkfCUDA.cu - Minkowski Functional pattern processing using CUDA NB: .cu assumes c++ style compilation
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors Jan-Oct 2019

#include "mkfCUDA.h"
#include "utilCUDA.hpp"

/***/

#include "bpfdCUDA.cuh"


/* HOWTO: Enque dependant jobs
   cudaStreamCreate( sk1, sk2 );
   cudaEventCreate( e[N] );
   for (i=0; i<N; i++)
   {
      k1<<<... sk1 >>>(...);
      cudaEventRecord(e[i], sk1);
      cudaStreamWaitEvent(sk2, e[i], 0);
      k2<<<... sk2 >>>(...);
   }
*/
extern "C"
size_t * mkfCUDAGetBPFD (KernInfo *pK, size_t * pBPFD, const BMOrg *pO, const BMPackWord * pW, const int profile)
{
   const int blkD= BPFD_BLKD;
   const int nBlk= (pO->rowPairs + blkD-1) / blkD;
   //LOG("\tsd= %u, %u\n", pO->rowWS, pO->planeWS);
   if (pBPFD)
   {
      CTimerCUDA t;//t.stampStream();
      switch (profile)
      {
         case MKFCU_PROFILE_FAST :
         {
            dim3 grd(nBlk,pO->planePairs,1);
            dim3 blk(blkD,1,1);
            addMultiPlaneSeqBPFD<<<grd,blk>>>((ULL*)pBPFD, pW, *pO);
            if (0 != ctuErr(NULL, "addMultiPlaneSeqBPFD"))
            { LOG(" .. <<<(%d,%d)(%d)>>>(%p, %p ..)\n", grd.x, grd.y, blk.x, pBPFD, pW); }
            break;
         }

         case MKFCU_PROFILE_FLEX :
         {
            dim3 grd(nBlk,pO->planePairs,1);
            dim3 blk(blkD,1,1);
            addPlaneBPFD<<<grd,blk>>>((ULL*)pBPFD, pW, pW + pO->planeWS, *pO); //, pBPFD+256);
            if (0 != ctuErr(NULL, "addPlaneBPFD"))
            { LOG(" .. <<<(%d,%d)(%d)>>>(%p, %p, %p ..)\n", grd.x, grd.y, blk.x, pBPFD, pW, pW + pO->planeWS); }
            break;
         }

         default :
         {
            for (int i= 0; i < pO->planePairs; i++)
            {
               const BMPackWord *pP0= pW + i * pO->planeWS;
               const BMPackWord *pP1= pW + (i+1) * pO->planeWS;
               //LOG(" RP: %d %d*%d=0x%08X, %p, %p \n", rowStride, planeStride, sizeof(*pP0), planeStride*sizeof(*pP0), pP0, pP1);
               addPlaneBPFD<<<nBlk,blkD>>>((ULL*)pBPFD, pP0, pP1, *pO); //, pBPFD+256);
               if (0 != ctuErr(NULL, "addPlaneBPFD"))
               { LOG(" .. <<<%d,%d>>>(%p, %p, %p ..)\n", nBlk, blkD, pBPFD, pP0, pP1); }
            }
            break;
         }
      }
      if (pK) { pK->dtms[1]= t.elapsedms(); }
      else { LOG("mkfCUDAGetBPFD() - dt= %Gms\n", t.elapsedms()); }
   }
   return(pBPFD);
} // mkfCUDAGetBPFD

typedef struct
{
   size_t *pH, *pD;
} HostDevPtrs;

// Global ptr to dev mem for lazy init hack
static HostDevPtrs gHD= { NULL, NULL };
#define BPFD_BYTES (sizeof(size_t)*MKF_BINS)

extern "C"
size_t * mkfCUDAGetBPFDH (KernInfo *pK, size_t * pBPFDH, const BMOrg *pO, const BMPackWord * pW, const int profile)
{
   if (NULL == pBPFDH)
   {
      if (NULL == gHD.pH) { cudaMallocHost(&(gHD.pH), BPFD_BYTES); }
      pBPFDH= gHD.pH;
   }
   if (NULL == gHD.pD) { cudaMalloc(&(gHD.pD), BPFD_BYTES); }
   if (pBPFDH && gHD.pD)
   {
      cudaMemset(gHD.pD, 0, BPFD_BYTES); // Kernel will add to BPFD so start clean
      if (mkfCUDAGetBPFD(pK, (size_t*)(gHD.pD), pO, pW, profile))
      {
         cudaMemcpy(pBPFDH, gHD.pD, BPFD_BYTES, cudaMemcpyDeviceToHost);
         return(pBPFDH);
      }
   }
   return(NULL);
} // mkfCUDAGetBPFDH

extern "C"
void mkfCUDACleanup (int lvl)
{
   if (gHD.pH) { cudaFreeHost(gHD.pH); gHD.pH= NULL; }
   if (gHD.pD) { cudaFree(gHD.pD); gHD.pD= NULL; }
   if (lvl > 0) { cudaDeviceReset(); } // NUKE!
} // mkfCUDACleanup

extern "C"
int mkfCUDAGetBPFDautoCtx (Context *pC, const int def[3], const BinMapF64 *pM, const int profHack)
{
   cudaError_t r;

   if (pC->pHF)
   {
      //r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice);

      if (NULL == pC->pDF)
      {
         r= cudaMalloc(&(pC->pDF), pC->bytesF);
         ctuErr(&r, "cudaMalloc()");
      }
      if (pC->pDF)
      {
         r= cudaMemcpy(pC->pDF, pC->pHF, pC->bytesF, cudaMemcpyHostToDevice);
         ctuErr(&r, "cudaMemcpy()");
      }

      if (NULL == pC->pDU)
      {
         r= cudaMalloc(&(pC->pDU), pC->bytesU);
         ctuErr(&r, "cudaMalloc()");
      }

      if (pC->pDF && pC->pDU)
      {
         ConstFieldPtr fieldPtr[BMCU_FIELD_MAX];
         BMFieldInfo fi;
         setupFields(&fi, autoFieldPtr(fieldPtr, pC), pC->nField, def, pC->bytesElem, profHack);
         if (NULL == binMapCUDA(&(pC->ki), pC->pDU, &(pC->bmo), &fi, pM))
         {
            LOG("\tpC= %p; pC->pDF= %p; &(pC->pDF)= %p\n\tpFDPT= %p; pFDPT->p= %p\n", pC, pC->pDF, &(pC->pDF),
               fi.pFieldDevPtrTable, fi.pFieldDevPtrTable->p);
            return(0);
         }
         else if (pC->pHU)
         {
            LOG("cudaMemcpy(%p, %p, %u)\n", pC->pHU, pC->pDU, pC->bytesU);
            r= cudaMemcpy(pC->pHU, pC->pDU, pC->bytesU, cudaMemcpyDeviceToHost);
            ctuErr(NULL, "{binMapCUDA+} cudaMemcpy()");
         }
      }
   }
   else if (pC->pHU)
   {
      r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice);
      if (NULL == pC->pDZ)
      {
         r= cudaMalloc(&(pC->pDZ), pC->bytesZ);
         ctuErr(&r, "cudaMalloc()");
      }
   }
   if (pC->pDU && pC->pDZ)
   {
      cudaMemset(pC->pDZ, 0, pC->bytesZ); // Kernel will add to BPFD so start clean
      mkfCUDAGetBPFD(&(pC->ki), (size_t*)(pC->pDZ), &(pC->bmo), pC->pDU, MKFCU_PROFILE_FAST);
      if (pC->pHZ)
      {
         LOG("cudaMemcpy(%p, %p, %u)\n", pC->pHZ, pC->pDZ, pC->bytesZ);
         r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost);
         ctuErr(&r, "{mkfCUDAGetBPFD+} cudaMemcpy()");
      }
   }
   return(1);
} // mkfCUDAGetBPFDautoCtx


// Internal test code
#ifdef MKF_CUDA_MAIN

#include "geomHacks.h"
#include "mkfUtil.h"

int buffAlloc (Context *pC, const int def[3], const NumEnc e, int flags)
{
   int b;

   pC->enc= e;
   pC->nElem= prodNI(def,3);
   pC->nField= 1; //MAX(1,n);
   pC->bytesF= encSizeN(&b, pC->nElem * pC->nField, e);
   if ((flags & 0x01) && (b > 0))
   {
      pC->bytesElem= BITS_TO_BYTES(b);
   }
   else { pC->nElem= 0; pC->nField= 0; pC->bytesElem= 0; pC->bytesF= 0; }

   if (flags & 0x02)
   {
      pC->nU= setBMO(&(pC->bmo), def, 0);
      pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   }
   else { pC->nU= 0; pC->bytesU= 0; }

   pC->nZ= MKF_BINS;
   pC->bytesZ= sizeof(size_t) * pC->nZ;

   LOG("F: %d * %d -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nField, pC->nElem, pC->bytesF, pC->nU, pC->bytesU);

   return cuBuffAlloc(pC,0);
} // buffAlloc

static const char gSepCh[2]={' ','\n'};

void dumpF (const float f[], const int n, const int wrap)
{
   int i=0;
   while (i<n)
   {
      int k= i + wrap;
      if (k > n) { k= n; }
      for (int j= i; j < k; j++) { LOG("%G%c", f[j], gSepCh[(j+1)==k]); }
      i= k;
   }
} // dumpF

void dumpUX (const uint u[], const int n, const int wrap)
{
   int i=0;
   while (i<n)
   {
      int k= i + wrap;
      if (k > n) { k= n; }
      for (int j= i; j < k; j++) { LOG("%08X%c", u[j], gSepCh[(j+1)==k]); }
      i= k;
   }
} // dumpUX

#if 0
checkHU ()
{
   size_t t= bitCountNU32(pC->pHU, pC->bytesU>>2);
   LOG("bitCountNU32() -> %zu\n", t);
   LOG("%p[%u]:\n",pC->pHU,pC->nU);
   m= def[0] >> BLKS; // def[0] / BLKD;
   n= m * def[1];
   if (n > pC->nU/2) { n= pC->nU/2; }
   while ((m<<1) < 16) { m<<= 1; }
   dumpUX(pC->pHU+2*n, n, m);
   LOG("%s\n","-");
   dumpUX(pC->pHU+3*n, n, m);
}
#endif
size_t mkftu (const Context *pC, const int def[3], const float mScale, const uint8_t profHack)
{
   cudaError_t r;
   size_t sum= 0;
   int verbose= 0;
   //KernInfo ki={0,0};

   r= cudaMemcpy(pC->pDU, pC->pHU, pC->bytesU, cudaMemcpyHostToDevice); ctuErr(&r, "cudaMemcpy(pDU, pHU)");
   r= cudaMemset(pC->pDZ, 0, pC->bytesZ); ctuErr(&r, "cudaMemset(pDZ)");
   mkfCUDAGetBPFD(NULL, (size_t*)(pC->pDZ), &(pC->bmo), pC->pDU, profHack);
   r= cudaMemcpy(pC->pHZ, pC->pDZ, pC->bytesZ, cudaMemcpyDeviceToHost); ctuErr(&r, "cudaMemcpy(pHZ, pDZ)");

   const size_t *pBPFD= (size_t*)pC->pHZ;
   float m[4];
   for (int i= 0; i<MKF_BINS; i++)
   {
      sum+= pBPFD[i];
      if (verbose && (pBPFD[i] > 0)) { LOG("[0x%X]=%u\n", i, pBPFD[i]); }
   }
   mkfRefMeasureBPFD(m, pBPFD, mScale);
   LOG("\tID%d: S=%zu\t%s: %G %G %G %G\n", profHack, sum, MKF_REFML, m[0],m[1],m[2],m[3]);

   return(sum);
} // mkftu

#define PAT_ID 3
int main (int argc, char *argv[])
{
   const int def[3]= {256, 256, 256}; // {64,64,2}; //{96,9,9};
   const float param[]= { def[0] * 0.75, 1, 0 }; //256-64;
   Context cux={0};
   const float mScale= 3.0 / sumNI(def,3); // reciprocal mean
   size_t n;
   char s[16];

   if (buffAlloc(&cux, def, ENC_F32, 0x03))
   {
      const size_t nC= prodOffsetNI(def,3,-1);
      LOG("[%d][%d][%d] -> %zuCells\n", def[0],def[1],def[2],nC);
      //sanityTest(&cux);
      if (cux.pHF)
      {
         BinMapF64 mc={0};//ENC_F32, 1,
         n= genPattern(NULL, cux.pHF, def, cux.enc, cux.nField, PAT_ID, param);
         LOG("genPattern(.. ENC_F32 ..) - n=%zu PCVF=%G\n", n, (float)n / cux.nElem);
         mkfCUDAGetBPFDautoCtx(&cux, def, setBinMapF64(&mc,">=",0.5), 0x00);
         const size_t *pBPFD= (size_t*)(cux.pHZ);
         float m[4];
         if (mkfSelectMeasuresBPFD(m, s, pBPFD, mScale, 2))
         {
            LOG("\t[F]:\t%s: %G %G %G %G\n", s, m[0],m[1],m[2],m[3]);
            LOG("\tbitCountNU32()=%zu\n", bitCountNU32(cux.pHU, cux.nU));
         }
      }
      if (cux.pHU)
      {
         memset(cux.pHU, 0, cux.bytesU);
#if 0    // word boundary test - should be factored into pattern gen
         const int wDef= cux.sd.row;
         const int lDef= def[1] * def[2];
         const uint m= BIT_MASK(def[0] & 0x1F);
         LOG("mkfCUDA - main() - [%d,%d,%d] -> [%d,%d]\n", def[0], def[1], def[2], wDef, lDef);
         for (int i= 0; i < lDef; i++)
         {  // NB: L to R order -> bits 0 to 31
            for (int j=0; j<wDef; j++) { cux.pHU[wDef * i + j]= 0xFFFFFFFF; }
            if (m) { cux.pHU[wDef * (i+1) - 1]= m; }
         }
         mkftu(&cux,def, mScale);

         int j= wDef/2;
         for (int i= 0; i < lDef; i++)
         {
            cux.pHU[wDef * i + j]= 0xFFF7FFFF;
         }
         for (int i= 0; i < lDef; i++)
         {
            int j= 0;
            for ( ; j<wDef-1; j++) { LOG("%08X ", cux.pHU[wDef * i + j]); }
            LOG("%08X\n", cux.pHU[wDef * i + j]);
         }
#endif
         if (0 != (def[0] & 0x1F)) { WARN("MKF_CUDA_MAIN / genPattern() - unpadded def[0]=%d\n", def[0]); }
         n= genPattern(NULL, cux.pHU, def, ENC_U1, 1, PAT_ID, param);
         LOG("genPattern(.. ENC_U1 ..) - n=%zu\n", n);
         mkftu(&cux,def,mScale,2);
         mkftu(&cux,def,mScale,3);
      }
      cuBuffRelease(&cux);
   }
   cudaDeviceReset();
} // main

#endif // MKF_CUDA_MAIN
