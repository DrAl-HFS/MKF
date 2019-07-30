
//#include "mkfTools.h"
#include "binMapAcc.h"
#include "mkfAcc.h"
#include "mkfCUDA.h"
#include "mkfUtil.h"
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

Bool32 buffAlloc (Context *pC, const int def[3])
{
   int r=0, vol= def[0] * def[1] * def[2];

   pC->nF= vol;
   pC->bytesF= sizeof(*(pC->pHF)) * pC->nF;
   pC->nU= BITS_TO_WRDSH(vol,5);
   pC->bytesU= sizeof(*(pC->pHU)) * pC->nU;
   pC->nZ= 256;
   pC->bytesZ= 8 * pC->nZ; // void * sizeof(*(pC->pHZ))

   LOG("F: %zu -> %zu Bytes\nU: %zu -> %zu Bytes\n", pC->nF, pC->bytesF, pC->nU, pC->bytesU);

#ifdef MKF_CUDA
   if (cuBuffAlloc(pC,vol)) { r= 2; }
#else
   pC->pHF= malloc(pC->bytesF);
   pC->pHU= malloc(pC->bytesU);
#endif
   if (pC->pHF) { memset(pC->pHF, 0, pC->bytesF); ++r; }
   if (pC->pHU) { memset(pC->pHU, 0xFF, pC->bytesU); ++r; }
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

void compareNZ (const size_t u0[], const size_t u1[], const int n)
{
   for (int i= 0; i < n; i++)
   {
      if (u0[i] != u1[i]) { LOG("[0x%X] %zu %zu\n", i, u0[i], u1[i]); }
   }
} // compareNZ

void symTst (void)
{
   U16 s;
   U8 m[1<<8];
   U8 s0, t0, n=0;

   for (int i=0; i<256; i++) { m[i]= i; }
   s= t0= 0x01;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 1;
   } while (s < 256);
   s= t0= 0x03;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 2;
   } while (s < 256);
   s= t0= 0x05;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 2;
   } while (s < 256);
   s= t0= 0x0A;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 2;
   } while (s < 256);
   s= t0= 0x0F;
   do
   {
      s0= s & 0xFF;
      m[s0]= t0;
      s0^= 0xFF;
      m[s0]= t0^0xFF;
      s<<= 4;
   } while (s < 256);
   for (int i=0; i<256; i++)
   {
      if (i != m[i]) { LOG("[%02X]=%02X\n", i, m[i]); n++; }
   }
   LOG("n=%u\n", n);
} // symTst

int main (int argc, char *argv[])
{
   const int def[3]= {256,256,256};
   BinMapF32 bmc;
   size_t aBPFD[256]={0,};
   Context cux={0};

   symTst();
   if (buffAlloc(&cux,def))
   {
      float vr= genPattern(cux.pHF, 4, def, 0.3*def[0] - 0.5);

      setBinMapF32(&bmc,">=",0.5);
      setupAcc();
      LOG("%smkfAccGetBPFDSimple() - \n", "***\n");
      mkfAccGetBPFDSimple(aBPFD, cux.pHU, cux.pHF, def, &bmc);
      LOG("\tvolFrac=%G (ref=%G) chiEP=%G (ref=%G)\n", volFrac(aBPFD), vr, chiEP3(aBPFD), 4 * M_PI);

#ifdef MKF_CUDA
      LOG("%smkfCUDAGetBPFDSimple() - \n", "***\n");
      if (mkfCUDAGetBPFDSimple(&cux, def, &bmc))
      {
         size_t *pBPFD= cux.pHZ;

         //pBPFD[0xFF]= aBPFD[0xFF]; // HAAACK!
         LOG("\tvolFrac=%G chiEP=%G\n", volFrac(pBPFD), chiEP3(pBPFD));
         //checkNZ(pBPFD, 256, "[%d]=%zu\n");
         //checkNZ(cux.pHU, cux.nU, NULL); // "[%d]: 0x%04X\n"
         compareNZ(aBPFD, pBPFD, MKF_BINS);
      }
#else
      //checkNZ(aBPFD, MKF_BINS, "[%d]=%zu\n");
#endif
   }
   buffRelease(&cux);
	return(0);
} // main
