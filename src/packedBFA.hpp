// packedBFA.hpp - Packed bit field array access
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#ifndef PACKED_BFA_HPP
#define PACKED_BFA_HPP

// Interface (abstract base) classes ( is this useful/necessary ? )
struct IWriteI { virtual void write (size_t idx, const int iVal) = 0; };

struct IReadI { virtual int read (size_t idx) const = 0; };


// Write PO2 (integral power-of-two i.e. 1,2,4.. bits) bit fields using
// RL field order (right to left i.e. "natural" shift order) within a
// 32bit word buffer. RL32 & PO2 offer simple and efficient implementation.
// NB: Non-PO2 involves word overlap, increasing complexity and
// requiring additional read-write cycles.
// NB: The alternate LR* field order is dominant in use for bitmap
// images. Historically, MSB-first transmission order is used in
// communication networks, so FAX images, memory buffers & thence
// file formats adopted the LR* ordering. Caveat Emptor.
class CWriteRL32P2B : public IWriteI
{
protected :
   uint8_t valS, valM, idxS, idxM; // NB: valM 8bit max!
   uint32_t *pW32; // union { uint32_t *pW32; float *pF32; };

public :

   CWriteRL32P2B (void *p=NULL, const uint8_t bits=1)
   {
      int l2= LOG2I(bits);
      if ((l2 >= 0) && (l2 <= 3) && (bits == (1 << l2)))
      {  // is valid integral power-of-two
         valS= l2;
         valM= BIT_MASK(bits);
         idxS= 5 - l2;
         idxM= BIT_MASK(idxS);
      }
      else { valS= valM= idxS= idxM= 0; }
      pW32= (uint32_t*)p;
   } // CWriteRL32P2B

   void write (size_t idx, const int iVal) override // NB: deliberately non-const...
   {  // although members unchanged, controlled buffer data is. (better pattern ?)
      uint8_t  iRL= (idx & idxM) << valS; // * bits
      // iLR= 32-iRL;
      uint32_t w32=  pW32[idx >>= idxS];  // Read word
      w32&= ~(valM << iRL);      // Mask target bits
      w32|= (iVal & valM) << iRL;// Assign value
      pW32[idx]= w32;   // Write back
   } // CWriteRL32P2B:: operator ()

}; // CWriteRL32P2B

// Extension with read interface for debug checks
class CReadWriteRL32P2B : public CWriteRL32P2B, IReadI
{
public :
   CReadWriteRL32P2B (void *p=NULL, const uint8_t bits=1) : CWriteRL32P2B(p,bits) { ; }

   // Inherits CWriteRL32P2B:: write ()

   int read (size_t idx) const override
   {
      uint8_t  iRL= (idx & idxM) << valS; // * bits
      uint32_t w32=  pW32[idx >>= idxS];  // Read word
      return((w32 >> iRL) & valM);      // Align and Mask value - NB: Not sign extended!
      //sign ext. = ((int)w32<<(32-(bits+iRL)) ) >> (32-bits);
   } // CReadWriteRL32P2B:: operator []

}; // CReadWriteRL32P2B


#endif // WRITER_HPP
