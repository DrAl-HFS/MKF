// cell8Sym.c - symmetry of 8 vertex (cubic) cell in 3-space.
// https://github.com/DrAl-HFS/MKF.git
// (c) Project Contributors June-August 2019

#include "cell8Sym.h"

// If the eight vertex locations of a cube are interpreted as a 3-tuple of 1bit coordinates
// labelled (X,Y,Z) Cartesian fashion then vertex locations correspond to three-bit binary
// numbers: 000, 001 ... 110, 111 equal to 0, 1 ... 6, 7 in decimal notation.
// The three rotations: RX(X,Y,Z) -> (X,Z,-Y), RY(X,Y,Z) -> (Z,Y,-X), RZ(X,Y,Z) -> (Y,X,-Z)
// and three reflections (one per axis) yield six basic symmetry-preserving transformations.
// This alone suggests ((2^3)*(2^3))-1 = 63 possible combinations, without considering the non-
// commutativity of 3D rotation. For any concatenated pair of such rotations (one or both of
// which may be a zero or null rotation) this suggests at most (3+1)^2=16 possibilities, but
// there is some redundancy therein (e.g. sequences of rotations about a single common axis).
// In the extreme case of six non-commutative transformations there would be :
//    !6 = 6!+5!+4!+3!+2!+1! = 873 total patterns.
// Happily, it seems that primitive rotations, in conjunction with all possible reflections,
// are sufficient to obtain the holomorphic group (homomorphs plus enantiomorphs) of a 3D cubic
// cell vertex pattern. There are at most 24 members of such a group, suggesting three rotation
// (one of which is null) times eight mirroring operations might be sufficient. However,
// determining which transformations are superfluous requires a difficult analysis of the
// starting pattern and so it is easier to evaluate 4*8=32 combined transformations, and then
// eliminate duplicates.

//const uint32_t gRRMask[]={0x11111111,0x22222222,0x44444444,0x33333333,0x55555555,0x66666666,0x77777777}; // X,Y,Z,XY,XZ,YZ,XYZ

typedef uint32_t PackV;

// Displace ?
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

static int genRotMasks (uint32_t m[3], const int nR, const int aR)
{
   int r=0;
   m[r++]= repBits(0x1, nR, aR);
   m[r++]= repBits(0x2, nR, aR);
   m[r++]= repBits(0x4, nR, aR);
   return(r);
} // genRotMasks

static int genMirMasks (uint32_t m[7], const int nR, const int aR)
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

static int patternToPackV (PackV r[1], uint8_t pattern, const int a)
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

static uint8_t packVToPattern (const PackV v[1], const int n, const int a)
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

static int mirrorPackVN (PackV r[], const PackV v, const uint32_t m[], const int nM)
{
   for (int i= 0; i < nM; i++) { r[i]= v ^ m[i]; }
   return(nM);
} // mirrorPackVN

static int mirrorPatternNoID (uint8_t r[], const uint8_t pattern)
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

static int rotatePatternNoID (uint8_t r[], const uint8_t pattern)
{
   const int align= 4;
   PackV v0;
   int nR= 0;
   int nB= patternToPackV(&v0, pattern, align);
   if (nB > 0)
   {
      PackV vR[3];
      uint32_t m[3];
      genRotMasks(m, nB, align);

      for (int axis= ROT_AXIS_Z; axis>= ROT_AXIS_X; axis--)
      {
         vR[nR]= rotatePackV(v0, axis, m);
         r[nR]= packVToPattern(vR+nR, nB, align);
         nR+= (r[nR] != pattern);
      }
   }
   return(nR);
} // rotatePatternNoID
/*    if (nB > 4) // unnecessary
      {  // Up to 9 composite rotations
         int nR0= nR;
         for (int i= 0; i<nR0; i++)
         {
            for (int axis= ROT_AXIS_Z; axis>= ROT_AXIS_X; axis--)
            {
               PackV tR= rotatePackV(vR[i], axis, m);
               r[nR]= packVToPattern(&tR, nB, align);
               nR+= (r[nR] != pattern);
            }
         }
      }*/

static int findPattern (const uint8_t v[], int n, const uint8_t pattern)
{
   for (int i= 0; i < n; i++)
   {
      if (pattern == v[i]) { return(i); }
   }
   return(-1);
} // findPattern

static int copyPatternUnique (uint8_t r[], int nR, const uint8_t v[], int n)
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

static void dump (uint8_t pat[], int n, const char *id, const char *fmt)
{
   LOG("%s[%d]=", id, n); for (int i=0; i < n; i++) { LOG(fmt, pat[i]); } LOG("%s", "\n");
} // dump

// Permute pattern geometry
static int permPatGeom (uint8_t v[], uint8_t u)
{
   int r, m, t, n= 0;
   int verbose= 0;

   v[n++]= u;
   if (verbose) { dump(v, n, "\nP", "0x%02X "); }

   r= rotatePatternNoID(v+n, v[n-1]);
   if (verbose) { dump(v+n,r," R", "0x%02X "); }
   t= n + r;

   m= mirrorPatternNoID(v+t, u);
   if (verbose) { dump(v+t, m, "  M", "0x%02X "); }
   t+= m;

   for (int i=0; i < r; i++)
   {
      m= mirrorPatternNoID(v+t, v[n+i]);
      if (verbose) { dump(v+t, m, "  M", "0x%02X "); }
      t+= m;
   }
   n= copyPatternUnique(v, n, v+n, t-n);
   return(n);
} // permPatGeom

int complementGroups (const int nC, const int nG, uint8_t patBuf[], GroupInf inf[])
{
   const int lG= nG+nC-1;
   int tC= 0;
   for (int i= 0; i<nC; i++)
   {
      const int iC= lG-i;
      inf[iC]= inf[i];
      inf[iC].bits= 8 - inf[iC].bits;
      tC+= inf[iC].count;
   }
   for (int i= 0; i<tC; i++) { patBuf[0xFF-i]= patBuf[i] ^ 0xFF; }
   return(nG+nC);
} // complementGroups

static const uint8_t gBasePat234[]=
{  // 0x00, 0x01 trivial base patterns
   0x03,0x09,0x18, // 2E "I" (d= R1, R2, R3)
   0x07,0x43,0x16, // 3E "L" (d= R1-R1, R1-R2, R2-R2)
   0x0F,0x17,0x27, // 4E "X" "T" "S" (d= R1*4)
   0x3C,0x69,0x87  // 4E "X" "?"
}; // Remaining patterns (5E-8E) are complements of 3E-0E
static const TFCounts gFEV234[]=
{  // HACKS!
   {1,4,4},{2,6,6},{4,12,8},
   {2,4,5},{3,7,7},{5,7,7},
   {4,4,4},{2,6,5},{4,7,5},
   {8,7,9},{9,8,11},{10,11,12},
};
static void setInf (GroupInf *pInf, uint8_t b, uint8_t n, TFCounts t)
   { pInf->bits= b; pInf->count= n; pInf->tfc= t; }

static uint edgeDM (const uint pattern, const uint m, const uint s)
{
   return((pattern & m) ^ ((pattern >> s) & m));
} // edgeDM

uint countEdgeV (uint pattern)
{
   uint m;
#if 1
   m= edgeDM(pattern, 0x0F, 4) << 16 |
      edgeDM(pattern, 0x33, 2) << 8 |
      edgeDM(pattern, 0x55, 1); // << 0;
#else
   m= (pattern & 0x0F) ^ ((pattern >> 4) & 0x0F) ;
   m|= ((pattern & 0x33) ^ ((pattern >> 2) & 0x33) ) << 8;
   m|= ((pattern & 0x55) ^ ((pattern >> 1) & 0x55) ) << 16;
#endif
   return bitCountZ(m);
} // countEdgeV

/* Interface */
int c8sGetPattern (uint8_t patBuf[CELL8_PATTERNS], GroupInf inf[CELL8_SYMM_GROUPS])
{
   TFCounts t={0,0,0};
   const int nBP= sizeof(gBasePat234);
   int iG=0, iPB= 0;
   int verbose= 0;

   if (verbose)
   {
      LOG_CALL("() [FP=%p]\n",__builtin_frame_address(0));
      LOG("\n%s\n", "Pattern Groups:");
      memset(patBuf,0x00,256);
   }

   patBuf[iPB]= 0x00; t.v= countEdgeV(patBuf[iPB]);
   setInf(inf+iG++, 0, 1, t);
   if (verbose) { dump(patBuf+0, 1, "G", "0x%02X "); LOG("v=%d\n", t.v); }
   iPB= 1;

   t.f= 1;
   t.e= 3;
   t.v= 3;
   for (int n= 0; n < 8; n++) { patBuf[iPB+n]= 1 << n; }
   t.v= countEdgeV(patBuf[iPB]);
   setInf(inf+iG++, 1, 8, t);
   if (verbose) { dump(patBuf+iPB, 8, "G", "0x%02X "); LOG("v=%d\n", t.v); }
   iPB+= 8;

   for (int iBP= 0; iBP < nBP; iBP++)
   {
      const uint8_t pattern= gBasePat234[iBP];
      const uint8_t b= bitCountZ(pattern);
      const uint8_t n= permPatGeom(patBuf+iPB, pattern);
      t.v= countEdgeV(pattern);
      setInf(inf+iG++, b, n, gFEV234[iBP]);
      if (verbose) { dump(patBuf+iPB, n, "G", "0x%02X "); LOG("v=%d\n", t.v); }
      iPB+= n;
   }

   return complementGroups(8, iG, patBuf, inf);
} // c8sGetPattern

int c8sGetMap (uint8_t groupMap[CELL8_PATTERNS])
{
   uint8_t patBuf[CELL8_PATTERNS];
   GroupInf inf[CELL8_SYMM_GROUPS];
   int nG= c8sGetPattern(patBuf, inf);
   if (nG > 0)
   {
      int n, k=0;
      for (int iG=0; iG<nG; iG++)
      {
         //b= inf[iG].bits;
         n= inf[iG].count;
         //LOG("[%2d]: {%d,%2d} (k=%d)\n", iG, b, n, k);
         for (int j=0; j<n; j++)
         {
            uint8_t pattern= patBuf[k+j];
            groupMap[ pattern ]= iG;
         }
         k+= n;
      }
   }
   return(nG);
} // c8sGetMap


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

void c8sTest (void)
{
   uint8_t groupMap[256];

   memset(groupMap,0xFF,256);
   c8sGetMap(groupMap);
   dump(groupMap, 256, "groupMap", "%d ");
#if 0
   {
      uint8_t r[256];
      //patBuf[0xFF]= patBuf[0x00];
      n= copyPatternUnique(r, 0, patBuf, 256);
   }
#endif
   //simpleTests();
} // c8sTest
