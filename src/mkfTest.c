
#include "mkfACC.h"
#include "mkfCUDA.h"
#include "geomHacks.h"
#ifdef __PGI
#include "openacc.h" // -> /opt/pgi/linux86-64/2019/include/openacc.h
// PGI: _DEF_OPENACC
// GNU: _OPENACC_H
#define OPEN_ACC_API
#endif

/***/

void setupAcc (void)
{  // Only multicore acceleration works presently: GPU produces garbage...
#ifdef OPEN_ACC_API
   LOG_CALL("() - %s\n", "acc_set_device_num( 0, acc_device_host )");
   acc_set_device_num( 0, acc_device_host );
#endif
} // setupAcc

B32 buffAlloc (Context *pC, const int def[3])
{
   //const int nC= prodOffsetNI(def, 3, -1);
   //const int lines= def[1]*def[2];
   int r=0;

   pC->nF= prodOffsetNI(def,3,0);
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= setBMSD(&(pC->sd), def, 0);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= MKF_BINS;
   pC->bytesZ= sizeof(size_t) * pC->nZ; // void * sizeof(*(pC->pHZ))

   LOG("F: %zu -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nF, pC->bytesF, pC->nU, pC->bytesU);

#ifdef MKF_CUDA
   if (cuBuffAlloc(pC,0)) { r= 2; }
#else
   pC->pHF= malloc(pC->bytesF);
   pC->pHU= malloc(pC->bytesU);
#endif
   if (pC->pHF) { memset(pC->pHF, 0, pC->bytesF); ++r; }
   if (pC->pHU) { memset(pC->pHU, 0xFF, pC->bytesU); ++r; }
   LOG("pHF=%p pHU=%p\n", pC->pHF, pC->pHU);
   return(r >= 2);
} // buffAlloc

void buffRelease (Context *pC)
{
#ifdef MKF_CUDA
   cuBuffRelease(pC);
#else
   if (NULL != pC->pHF) { free(pC->pHF); pC->pHF= NULL; }
   if (NULL != pC->pHU) { free(pC->pHU); pC->pHU= NULL; }
#endif
} // buffRelease

void checkNZ (const size_t u[], const int n, const char *pVrbFmt)
{ // debug...
   size_t t[2]= { 0, 0 };
   for (int i= 0; i < n; i++)
   {
      if (pVrbFmt && (0 != u[i])) { LOG(pVrbFmt, i, u[i]); }
      t[0]+= u[i] * bitCountZ(i);
      t[1]+= bitCountZ(u[i]);
   }
   LOG_CALL("(.. %d ..) - bitcounts: dist=%zu /8= %zu, raw=%zu\n", n, t[0], t[0]>>3, t[1]);
} // checkNZ

int compareNZ (const size_t u0[], const size_t u1[], const int n, const int flags)
{
   int d, sDiff=0, nDiff= 0;
   for (int i= 0; i < n; i++)
   {
      d= (int)(u1[i] - u0[i]);
      sDiff+= d;
      nDiff+= 0 != d;
      if (flags && d) { LOG("[0x%X] %zu %zu\n", i, u0[i], u1[i]); }
   }
   LOG("compareNZ() - sDiff=%d, nDiff=%d\n", sDiff, nDiff);
   return(nDiff);
} // compareNZ

int main (int argc, char *argv[])
{
   int id=4, def[3]= {256+1,256, 256}; //256};
   BinMapF32 bmc;
   size_t aBPFD[256]={0,}, aBPFD2[256]={0,};
   Context cux={0};

   //c8sTest();
   //geomTest(2,2);
   mkfuTest();
   //printf("long int = %dbytes\n", sizeof(long int));
   if (buffAlloc(&cux,def))
   {
      const float param= 256-64; //midRangeHNI(def,3)-3;
      float vfR= genPattern(cux.pHF, id, def, param);
      float m[4];

      if (vfR <= 0) { WARN("genPattern() - vfR=%G\n", vfR); }

      setBinMapF32(&bmc,">=",0.5);
      setupAcc();
      LOG("%smkfAccGetBPFDSimple() - \n", "***\n");
      mkfAccGetBPFDSimple(aBPFD, cux.pHU, cux.pHF, def, &bmc);
      if (refMeasures(m, aBPFD, 1.0/257))
      {
         LOG(" refMeasures() - V S M K: %G %G %G %G\n", m[3],m[2],m[1],m[0]);
      }
      //LOG("\tvolFrac=%G (ref=%G, vf8=%G) chiEP=%G (ref=%G)\n", volFrac(aBPFD), vfR, volFrac8(aBPFD), chiEP3(aBPFD), 4 * M_PI);


      SWAP(int,def[0],def[2]);
      vfR= genPattern(cux.pHF, id, def, param);
      mkfAccGetBPFDSimple(aBPFD2, cux.pHU, cux.pHF, def, &bmc);
      if (refMeasures(m, aBPFD2, 1.0/257))
      {
         LOG(" refMeasures() - V S M K: %G %G %G %G\n", m[3],m[2],m[1],m[0]);
      }
      compareNZ(aBPFD, aBPFD2, MKF_BINS, 0x0);

#ifdef MKF_CUDA
      LOG("mkfCUDAGetBPFDautoCtx() - %G\n", bmc.t[0]);
      if (mkfCUDAGetBPFDautoCtx(&cux, def, &bmc))
      {
         size_t *pBPFD= cux.pHZ;

         //pBPFD[0xFF]= aBPFD[0xFF]; // HAAACK!
         LOG("\tvolFrac=%G chiEP=%G\n", volFrac(pBPFD), chiEP3(pBPFD));
         //checkNZ(pBPFD, 256, "[%d]=%zu\n");
         //checkNZ(cux.pHU, cux.nU, NULL); // "[%d]: 0x%04X\n"
         compareNZ(aBPFD, pBPFD, MKF_BINS, 1);
      }
#else
      //checkNZ(aBPFD, MKF_BINS, "[%d]=%zu\n");
#endif
   }
   buffRelease(&cux);
	return(0);
} // main
