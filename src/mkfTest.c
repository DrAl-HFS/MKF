
#include "mkfTools.h"
#include "binMap.h"
#include "mkf.h"

/*  */

// -> GEOMETRY ?
F32 sphereVol (F32 r) { return((4 * M_PI / 3) * r * r *r); }

F32 boxVol (F32 r) { return(8 * r * r * r); }

F32 mag2 (F32 dx, F32 dy, F32 dz) { return(dx*dx + dy*dy + dz*dz); }

/***/

int genBall (F32 *pF, const int def[3], const F32 r)
{
   size_t i= 0, n= 0;
   F32 c[3], r2= r * r;

   for (int d=0; d<3; d++) { c[d]= 0.5 * def[d]; }

   for (int j= 0; j<def[2]; j++)
   {
      for (int k= 0; k<def[1]; k++)
      {
         for (int l= 0; l<def[0]; l++)
         {
            if (mag2(j-c[0], k-c[1], l-c[2]) <= r2) { pF[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBall

int genBox (F32 *pF, const int def[3], const F32 r)
{
   size_t i= 0, n= 0;
   F32 c[3];

   for (int d=0; d<3; d++) { c[d]= 0.5 * def[d]; }

   for (int j= 0; j<def[2]; j++)
   {
      for (int k= 0; k<def[1]; k++)
      {
         for (int l= 0; l<def[0]; l++)
         {
            if ( (abs(j-c[2]) <= r) && (abs(k-c[1]) <= r) && (abs(l-c[0]) <= r) ) { pF[i]= 1.0; ++n; }
            ++i;
         }
      }
   }
   return(n);
} // genBox

extern Bool32 mkfProcess (Context *pC, const int def[3], const MKBMapF32 *pMC);

int main (int argc, char *argv[])
{
   size_t n, vol;
   F32 *pF;
   U32 *pBM;
   const int def[3]= {64,64,64};
   const F32 radius= 0.5*def[0] - 1.5;
   F32 fracR;
   BinMapF32 bmc;
   U32 hBPD[256]={0,};
   MKMeasureVal vf, vr, kf;

   vol= def[0] * def[1] * def[2];
   n= sizeof(*pF)*vol;
   LOG("vol= %zu sites\nF: %zu Bytes\n", vol, n);
   pF= malloc(n);
   if (pF)
   {
      memset(pF, 0, n);
      n= BITS_TO_BYTES(def[0]) * def[1] * def[2];
      LOG("BM: %zu Bytes\n", n);
      pBM= malloc(n);
      if (pBM)
      {
         memset(pBM, 0xFF, n);

         fracR= radius / def[0];
#if 1
         vr= sphereVol(fracR);
         n= genBall(pF, def, radius);
         LOG("ball=%zu (/%d=%G)\n", n, vol, (F64)n / vol);
#else
         vr= boxVol(fracR);
         n= genBox(pF, def, radius);
         LOG("box=%zu (/%d=%G)\n", n, vol, (F64)n / vol);
#endif
         setBMCF32(&bmc,">=",0.5);
         procSimple(hBPD, pBM, pF, def, &bmc);
         { // debug...
            size_t t= 0;
            for (int i= 0; i < 256; i++)
            {
               //LOG("%d: %u\n", i, hBPD[i]);
               t+= hBPD[i] * bitCountZ(i);
            }
            LOG("bitcount=%zu /8= %zu\n", t, t>>3);
         }

         vf= volFrac(hBPD);
         kf= chiEP3(hBPD);
         LOG("volFrac=%G (ref=%G)\n", vf, vr);
         LOG("chiEP=%G (ref=%G)\n", kf, 4 * M_PI);
#ifdef MK_CUDA
         {
            Context ctx={0};
            ctx.pHF= pF; ctx.nF= vol; ctx.bytesF= ctx.nF * sizeof(*pF);
            ctx.pHU= pBM; ctx.nU= ctx.nF / 32; ctx.bytesU= ctx.nU * sizeof(*pBM);
            ctx.nZ= 256; ctx.bytesZ= ctx.nZ * sizeof(uint); // ctx.pHZ= (void*)cudaMallocHost(ctx.bytesZ);
            if (mkfProcess(&ctx, def, &bmc))
            {
               const uint *pBPD= ctx.pHZ;
               LOG("mkfProcess() - volFrac=%G chiEP=%G\n", volFrac(pBPD), chiEP3(pBPD));
            }
         }
#endif
         free(pBM);
      }
      free(pF);
   }
   //utilTest();
	//LOG_CALL("() %s\n", "MKF");
	return(0);
} // main
