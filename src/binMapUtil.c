// binMapUtil.c - packed binary map general utility code.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June 2019

#include "binMapUtil.h"


/***/

#if 0
// '<' '=' '>' -> -1 0 +1
//int sgnF32 (const F32 f) { return((f > 0) - (f < 0)); }

// '<' '=' '>' -> 0, 1, 2
U32 sgnIdxDiff1F32 (const F32 f, const F32 t) { return(1 + (f > t) - (f < t)); } // 1+sgnF32(f-t)

U32 bm1F32 (const F32 f, const BinMapF32 * const pC)
{  // pragma acc data present( pC[:1] )
   return( (pC->m >> sgnIdxDiff1F32(f, pC->t[0]) ) & 0x1 );
} // bm1F32

// interval threshold (hysteresis)
U32 sgnIdxDiff2F32 (const F32 f, const F32 t[2])
{
   return( sgnIdxDiff1F32(f,t[0]) + 3 * sgnIdxDiff1F32(f,t[1]) );
} // sgnIdxDiff2F32

U8 bm2F32 (const F32 f, const BinMapCtxF32 *pC)
{  // pragma acc data present( pC[:1] )
   return( ( pC->m >> (sgnIdxDiff2F32(f, pC->t[0]) + 3 * sgnIdxDiffF32(f, pC->t[1])) ) & 0x1 );
} // bm2F32

#endif

/***/

BinMapF32 *setBinMapF32 (BinMapF32 *pC, const char relopChars[], const float t)
{
   int i= 0, inv= 0, valid= 1;
   pC->t[0]= t;
   pC->m= BMC_NV;

   do
   {
      switch(relopChars[i])
      {
         case '<' : pC->m|= BMC_LT; break;
         case '=' : pC->m|= BMC_EQ; break;
         case '>' : pC->m|= BMC_GT; break;
         case '!' : if (0 == i) { inv= 1; break; } // else...
         default : valid= 0;
      }
      ++i;
   } while (valid);
   if (inv) { pC->m^= BMC_AL; }
   //LOG_CALL("(%p, %s, %G) - m=0x%X\n", pC, relopChars, t, pC->m);
   return(pC);
} // setBinMapF32

size_t setBMO (BMOrg *pO, const int def[3], const char profID)
{
   BMStride rowStride, planeStride;
   uint maxRow= MIN(2, def[1]);
   uint maxPlane= MIN(2, def[2]);

   rowStride= BITS_TO_WRDSH(def[0],5);  // Packed 32bit words
   switch (profID & 0x0F)
   {
      case 0 :
         maxRow=   def[1];
         maxPlane= def[2];
         break;
   }
   planeStride= rowStride * maxRow;
   if (pO)
   {
      pO->rowElem= def[0];
      pO->rowPairs= maxRow-1;
      pO->planePairs= maxPlane-1;
      pO->rowWS=   rowStride;
      pO->planeWS= planeStride;
   }
   return((size_t)planeStride * maxPlane);
} // setBMSD

int copyNonZeroDef (FieldDef d[], const FieldDef *pD, const int nD)
{
   int rD= 0;
   if (pD)
   {
      for (int i=0; i<nD; i++)
      {
         d[rD]= pD[i];
         rD+= (d[rD] > 1);
      }
   }
   FieldDef pad= (rD>0);
   for (int i=rD; i<nD; i++) { d[i]= pad; }
   return(rD);
} // copyNonZeroDef

int validPtr (const void *p) { return(NULL != p); } // Type code, address range ??

int copyValidPtrByMask (ConstFieldPtr r[], const int max, const ConstFieldPtr a[], const uint mask)
{
   int t, i= 0, n= 0;
   do
   {
      t= (0x1 << i);
      if (validPtr(a[i].p) && (mask & t))
      {
         r[n++]= a[i];
      }
      i++;
   } while ((mask > t) && (n < max));
   //if (n < max) { r[n].p= NULL; } guard ?
   return(n);
} // copyValidPtrByMask

int countValidPtrByMask (const ConstFieldPtr a[], uint mask)
{
   int i= 0, n= 0;
   do
   {
      n+= validPtr(a[i].p) && (mask & 0x1);
      mask >>= 1;
      i++;
   } while (mask > 0);
   return(n);
} // copyValidPtrByMask

int genStride (FieldStride fs[], const int n, const int start, const FieldDef *pD, FieldStride stride)
{
   int i, r= 0;
   if (pD && (0 != stride))
   {  // Generate stride
      for (i= 0; i<start; i++) { stride*= pD[i]; }
      while (r < n) { fs[r++]= stride; stride*= pD[i]; }
   }
   return(r);
} // genStride

int copyOrGenStride (FieldStride fs[], const int n, const int start, const FieldDef *pD, const FieldStride *pS)
{
   int nR= 0;
   if (pS)
   {
      for (int i=0; i<n; i++) { fs[nR++]= pS[i+start]; }
   }
   else { nR= genStride(fs, n, start, pD, 1); }
   return(nR);
} // copyOrGenStride


ConstFieldPtr *asFieldTab (const void **pp, const NumEnc id)
{
   ConstFieldPtr *pFP= (ConstFieldPtr*)pp;
   int a= encAlignment(id);
   if (0 != (pFP->w & (a-1))) { WARN("[mkfCUDA] alignment %u %p\n", a, pFP->p); }
   return(pFP);
} // asFieldTab

const BMFieldInfo *setupFields (BMFieldInfo *pI, void **ppF, const int n, const int def[3], const int b, const int profile)
{  // field arg should be const float **ppF, but pgc++ appears to have defective const-ness rule for multiple indirection
   if (pI && (n > 0) && (n < 32))
   {
      switch(b)
      {
         case sizeof(float) : pI->elemID= ENC_F32; break;
         case sizeof(double) : pI->elemID= ENC_F64; break;
         default : return(NULL);
      }
      pI->fieldTableMask= BIT_MASK(n);
      pI->oprID= 0;
      pI->profID= profile;
      pI->flags= 0;
      pI->pD= def;
      pI->pS= NULL;  // NULL => assume fully planar fields
      pI->pFieldDevPtrTable= asFieldTab((const void**)ppF, pI->elemID);
      return(pI);
   }
   return(NULL);
} // setupFields

void **autoFieldPtr (ConstFieldPtr ptr[BMCU_FIELD_MAX], const Context *pC)
{
   if (pC->nField > 1)
   {
      const FieldStride stride= pC->nElem * pC->bytesElem;
      ConstFieldPtr t={pC->pDF};//t.p= pC->pDF;

      int max= MIN(BMCU_FIELD_MAX, pC->nField);
      for (int i=0; i<max; i++)
      {
         ptr[i].w= t.w;
         t.w+= stride;
      }
      return((void**)ptr); // &((void*)(ptr[0].p));
   }
   //else
   return((void**)&(pC->pDF));
} // autoFieldPtr


#if 0
testBMC (const float f0, const float fs, const int n, const BinMapF32 *pC)
{
   for (int i= 0; i<n; i++)
   {
      F32 f= f0 + i * fs;
      U8 r= bm1F32(f, &ctx);
      LOG("\t%G -> %u\n", f, r);
   }
}
#endif

