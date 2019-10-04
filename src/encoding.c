
#include "encoding.h"


// Return storage size in bytes of n elements, optionally set bit size of single element
size_t encSizeN (int *pElemBits, const size_t n, const NumEnc e)
{
   size_t bytes;
   int b= e & ENC_MASK_NUM;
   switch(e & ENC_MASK_TYP)
   {
      case ENC_TYPE_FPB :
      case ENC_TYPE_INT :
         bytes= b * n; b<<= 3;
         break;
      default : WARN_CALL("(.. 0x%02X) - assuming bit encoding\n", e);
      case ENC_TYPE_BTS   : bytes= (b * n) >> 3; // Packed!
         break;
   }
   if (pElemBits) { *pElemBits= b; }
   return(bytes);
} // encSizeN

int encAlignment (const NumEnc e)
{
   int b= e & ENC_MASK_NUM;
   switch(e & ENC_MASK_TYP)
   {
      case ENC_TYPE_FPB :
      case ENC_TYPE_INT :
         return(b);
         //break;
      default : WARN_CALL("(.. 0x%02X) - assuming bit encoding\n", e);
      case ENC_TYPE_BTS   : return BITS_TO_BYTES(b); // Packed!
         //break;
   }
} // encAlignment
