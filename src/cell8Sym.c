// cell8Sym.c - symmetry of 8 vertex (cubic) cell in 3-space.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#include "cell8Sym.h"

// If the eight vertex locations of a cube are interpreted as a 3-tuple of 1bit coordinates
// labelled (X,Y,Z) Cartesian fashion then vertex locations correspond to three-bit binary
// numbers: 000, 001 ... 110, 111 equal to 0, 1 ... 6, 7 in decimal notation.
// The three rotations: (X,Y,Z) -> (X,Z,Y), (Z,Y,X), (Y,X,Z) and three reflections (one per axis)
// yield six basic symmetry-preserving transformations. This suggests (2^6)-1 = 63 combinations
// but does not account for the non-commutativity of 3D rotation. As reflection is commutative
// and repetitions cancel there are only seven combinations - these are easily dealt with.
//
// (In the extreme case !6 = 6!+5!+4!+3!+2!+1! = 873 total patterns.)

//const uint32_t gRRMask[]={0x11111111,0x22222222,0x44444444,0x33333333,0x55555555,0x66666666,0x77777777}; // X,Y,Z,XY,XZ,YZ,XYZ

typedef uint32_t PackV;

// Args: basic mask, number of repetitions, alignment
uint32_t repBits (uint32_t m, const int nR, const int aR)
{
   uint32_t r=0;

   for (int8_t i=0; i<nR; i++)
   {
      r|= m;
      m<<= aR;
   }
   return(r);
} // repBits

int genRotMasks (uint32_t m[3], const int nR, const int aR)
{
   int r=0;
   m[r++]= repBits(0x1, nR, aR);
   m[r++]= repBits(0x2, nR, aR);
   m[r++]= repBits(0x4, nR, aR);
   return(r);
} // genRotMasks

int genMirMasks (uint32_t m[7], const int nR, const int aR)
{
   int r= genRotMasks(m, nR, aR);
   //if (nM >= 6)
   m[r++]= m[0] | m[1];
   m[r++]= m[0] | m[2];
   m[r++]= m[1] | m[2];
   //if (nM >= 7)
   m[r++]= m[0] | m[1] | m[2];
   return(r);
} // genMirMasks

#define ROT_AXIS_BITS (2)
#define ROT_AXIS_MASK BIT_MASK(ROT_AXIS_BITS)
#define ROT_AXIS_X (0)
#define ROT_AXIS_Y (1)
#define ROT_AXIS_Z (2)
#define ROT_AXIS_N (3)

// Args: patternV, rotation-axis index, RRmask
PackV rotatePackV (PackV v, int cmd, const uint32_t m[3])
{
   switch(cmd & ROT_AXIS_MASK)
   {
      case ROT_AXIS_X : return (v & m[0]) | ((v & m[1]) << 1) | (((v^m[2]) & m[2]) >> 1); // break; // X,Y,Z -> X,-Z,Y
      case ROT_AXIS_Y : return (v & m[1]) | ((v & m[0]) << 2) | (((v^m[2]) & m[2]) >> 2); // break; // X,Y,Z -> -Z,Y,X
      case ROT_AXIS_Z : return (v & m[2]) | ((v & m[0]) << 1) | (((v^m[1]) & m[1]) >> 1); // break; // X,Y,Z -> -Y,X,Z
   }
   return(v);
} // rotatePackV

// Args: results, patternV, rotation-axis indices, number of rotations, RRmask
PackV rotateNRPackV (PackV r[], PackV v, int cmdSeq, const int nCmd, const uint32_t m[3])
{
   int cumulative= cmdSeq & (1<<31);
   for (int i= 0; i < nCmd; i++)
   {
      switch(cmdSeq & ROT_AXIS_MASK)
      {
         case ROT_AXIS_X : r[i]= (v & m[0]) | ((v & m[1]) << 1) | (((v^m[2]) & m[2]) >> 1); break; // X,Y,Z -> X,-Z,Y
         case ROT_AXIS_Y : r[i]= (v & m[1]) | ((v & m[0]) << 2) | (((v^m[2]) & m[2]) >> 2); break; // X,Y,Z -> -Z,Y,X
         case ROT_AXIS_Z : r[i]= (v & m[2]) | ((v & m[0]) << 1) | (((v^m[1]) & m[1]) >> 1); break; // X,Y,Z -> -Y,X,Z
      }
      cmdSeq>>= ROT_AXIS_BITS;
      if (cumulative) { v= r[i]; }
   }
   return(nCmd);
} // rotateNRPackV

int patternToPackV (PackV r[1], uint8_t pattern, const int a)
{
   int i= 0, n= 0, s= 0;
   uint32_t packV= 0;
   while ((pattern > 0) && (i < 8))
   {
      if (pattern & 0x1) { packV |= i << s; s+=a; n++; }
      pattern>>= 1;
      i++;
   }
   r[0]= packV;
   return(n);
} // patternToPackV

uint8_t packVToPattern (const PackV v[1], const int n, const int a)
{
   const int mA= BIT_MASK(a);
   uint8_t pattern=0;
   uint32_t packV= v[0];
   for (int i= 0; i < n; i++)
   {
      pattern|= 1 << (packV & mA);
      packV >>= a;
   }
   return(pattern);
} // packVToPattern

/***/

int mirrorPackVN (PackV r[], const PackV v, const uint32_t m[], const int nM)
{
   for (int i= 0; i < nM; i++) { r[i]= v ^ m[i]; }
   return(nM);
} // mirrorPackVN

int mirrorPatternNoID (uint8_t r[], const uint8_t pattern)
{
   const int align= 4;
   PackV v0;
   int nR= 0;
   int nB= patternToPackV(&v0, pattern, align);
   if (nB > 0)
   {
      uint32_t m[7];
      genMirMasks(m, nB, align);

      for (int i= 0; i < 7; i++)
      {
         PackV vM= v0 ^ m[i];
         r[nR]= packVToPattern(&vM, nB, align);
         nR+= (r[nR] != pattern);
      }
   }
   return(nR);
} // mirrorPatternNoID

int rotatePatternNoID (uint8_t r[], const uint8_t pattern)
{
   const int align= 4;
   PackV v0;
   int nR= 0;
   int nB= patternToPackV(&v0, pattern, align);
   if (nB > 0)
   {
      uint32_t m[3];
      genRotMasks(m, nB, align);

      for (int axis= ROT_AXIS_X; axis<= ROT_AXIS_Z; axis++)
      {
         PackV vR= rotatePackV(v0, axis, m);
         r[nR]= packVToPattern(&vR, nB, align);
         nR+= (r[nR] != pattern);
      }
   }
   return(nR);
} // rotatePatternNoID

int findPattern (const uint8_t v[], int n, const uint8_t pattern)
{
   for (int i= 0; i < n; i++)
   {
      if (pattern == v[i]) { return(i); }
   }
   return(-1);
} // findPattern

int copyPatternUnique (uint8_t r[], int nR, const uint8_t v[], int n)
{
   for (int i= 0; i < n; i++)
   {
      if (findPattern(r, nR, v[i]) < 0)
      {
         r[nR++]= v[i];
      }
   }
   return(nR);
} // copyPatternUnique


void dump (uint8_t pat[], int n, const char *id)
{
   LOG("%s[%d]=", id, n); for (int i=0; i < n; i++) { LOG("0x%02X ", pat[i]); } LOG("%s", "\n");
} // dump

// Permute pattern geometry
int permPatGeom (uint8_t v[], uint8_t u)
{
   int r, m, t, n= 0;

   v[n++]= u;
   dump(v, n, "\nP");
   r= rotatePatternNoID(v+n, v[n-1]);
   dump(v+n,r," R");
   t= n + r;
   for (int i=0; i < r; i++)
   {
      m= mirrorPatternNoID(v+t, v[n+i]);
      dump(v+t, m, "  M");
      t+= m;
   }
   n= copyPatternUnique(v, n, v+n, t-n);
   return(n);
} // permPatGeom


static const uint8_t gBasePat[]=
{  // 0x00, 0x01
   0x03,0x09,0x81, // 2E (R1, R2, R3)
   0x07,0x43,0x83,  // 3E "L"
   0x0F,0xC3       // 4E
};

void c8sTest (void)
{
   uint8_t patBuf[256]; //, tmp[16];
   LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));

   int t= 0, n= 0, nBP;

   LOG("\n%s\n", "Pattern Groups:");

   patBuf[n++]= 0x00; dump(patBuf+t, n, "G");
   t+= n;

   for (n= 0; n < 8; n++) { patBuf[t+n]= 1 << n; } dump(patBuf+t, n, "G");
   t+= n;

   nBP= sizeof(gBasePat);
   for (int iBP= 0; iBP < nBP; iBP++)
   {
      n= permPatGeom(patBuf+t, gBasePat[iBP]);
      dump(patBuf+t, n, "G");
      t+= n;
   }

   LOG("\nt=%d\n***\n", t);
   //simpleTests();
} // c8sTest


void simpleTests (void)
{
   const int align= 4;
   uint32_t rrm[7];
   uint32_t packV[64];
   uint8_t pattern[16];
   int nB=2, nV=0, tV=0;

   LOG("\n%s\n", "genRRMasks() :");
   genMirMasks(rrm, 8, align);
   for (int i=0; i<7; i++) { LOG(" %x", rrm[i]); }

   LOG("\n%s\n", "Pack/Unpack");
   for (int i=0; i < 8; i++)
   {
      uint8_t t= 1 << i;
      nB= patternToPackV(packV+nV, t, align);
      pattern[nV]= packVToPattern(packV+nV, nB, align);
      LOG("0x%02x (%d) : 0x%08x 0x%02x\n", t, nB, packV[nV], pattern[nV]);
      nV++;
   }

   LOG("\n%s\n", "Mirror");
   nB= patternToPackV(packV+nV, 0x03, align);
   genMirMasks(rrm, nB, align);

   nV= (nB > 0);
   tV= nV;
   tV+= mirrorPackVN(packV+nV, packV[0], rrm, 7);
   for (int i=0; i < tV; i++)
   {
      pattern[i]= packVToPattern(packV+i, nB, align);
      LOG("0x%08x 0x%02x\n", packV[i], pattern[i]);
   }

   LOG("\n%s\n", "Rotate");
   tV= nV;
   // Need permutations...
   //packV[tV++]= rotatePackV(packV[0], ROT_AXIS_X, rrm);
   //packV[tV++]= rotatePackV(packV[0], ROT_AXIS_Y, rrm);
   //packV[tV++]= rotatePackV(packV[0], ROT_AXIS_Z, rrm);
   int cmd= ROT_AXIS_Z | (ROT_AXIS_Y<<2) | (ROT_AXIS_X<<4);
   tV+= rotateNRPackV(packV+tV, packV[0], cmd, 3, rrm);

   for (int i=0; i < tV; i++)
   {
      pattern[i]= packVToPattern(packV+i, nB, align);
      LOG("0x%08x 0x%02x\n", packV[i], pattern[i]);
   }
} // simpleTests
