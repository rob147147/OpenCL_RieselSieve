ulong2 xbinGCD(ulong a, ulong b){
    ulong alpha, beta, u, v;
    ulong2 out;
    u = 1; 
    v = 0;
    alpha = a; 
    beta = b; 
    // Note that alpha is even and beta is odd.
    /* The invariant maintained from here on is: a = u*2*alpha - v*beta. */
 
    while (a > 0) {
        a = a >> 1;
        if ((u & 1) == 0) { 
            u = u >> 1; 
            v = v >> 1; 
        } 
        else {
            /* We want to set u = (u + beta) >> 1, but that can overflow, so we use Dietz's method. */
            u = ((u ^ beta) >> 1) + (u & beta);
            v = (v >> 1) + alpha;
        }
    }
    out.x = u;
    out.y = v;
 return out;
} 

ulong modul64(ulong x, ulong y, ulong z) {
    /* Divides (x || y) by z, for 64-bit integers x, y, and z, giving the remainder (modulus) as the result. Must have x < z (to get a 64-bit result). This is checked for. */
    long t;

    for (int i = 1; i <= 64; i++) { // Do 64 times.
        t = (long)x >> 63; // All 1's if x(63) = 1.
        x = (x << 1) | (y >> 63); // Shift x || y left
        y = y << 1; // one bit.
        if ((x | t) >= z) {
            x = x - z;
            y = y + 1;
        }
    }
 return x; // Quotient is y.
}

uint8 mul24_120(uint8 a, uint8 b, uint8 res) {
    //Multiply two 64 bit ulongs using mul24, add a third ulong and return the low 120 bits. 
 
    res.s7 = mad24(a.s7,b.s7,res.s7);

    res.s6 = res.s6 + mad24(a.s6,b.s7,res.s7>>15);
    res.s6 = mad24(a.s7,b.s6,res.s6);

    res.s5 = res.s5 + mad24(a.s5,b.s7,res.s6>>15);
    res.s5 = mad24(a.s6,b.s6,res.s5);
    res.s5 = mad24(a.s7,b.s5,res.s5);

    res.s4 = res.s4 + mad24(a.s4,b.s7,res.s5>>15);
    res.s4 = mad24(a.s5,b.s6,res.s4);
    res.s4 = mad24(a.s6,b.s5,res.s4);
    res.s4 = mad24(a.s7,b.s4,res.s4);

    res.s3 = res.s3 + mad24(a.s3,b.s7,res.s4>>15);
    res.s3 = mad24(a.s4,b.s6,res.s3);
    res.s3 = mad24(a.s5,b.s5,res.s3);
    res.s3 = mad24(a.s6,b.s4,res.s3);
    res.s3 = mad24(a.s7,b.s3,res.s3);

    res.s2 = res.s2 + mad24(a.s3,b.s6,res.s3>>15);
    res.s2 = mad24(a.s4,b.s5,res.s2);
    res.s2 = mad24(a.s5,b.s4,res.s2);
    res.s2 = mad24(a.s6,b.s3,res.s2);

    res.s1 = res.s1 + mad24(a.s3,b.s5,res.s2>>15);
    res.s1 = mad24(a.s4,b.s4,res.s1);
    res.s1 = mad24(a.s5,b.s3,res.s1);

    res.s0 = res.s0 + mad24(a.s3,b.s4,res.s1>>15);
    res.s0 = mad24(a.s4,b.s3,res.s0);

    return res;
}

uint8 mul24_64(uint8 a, uint8 b) {
    //Return the low half of multiplying two 64 bit ulongs using mul24. 

    uint8 res = {0,0,0,0,0,0,0,0};

    res.s7 = mul24(a.s7,b.s7);

    res.s6 = mad24(a.s6,b.s7,res.s7>>15);
    res.s6 = mad24(a.s7,b.s6,res.s6);

    res.s5 = mad24(a.s5,b.s7,res.s6>>15);
    res.s5 = mad24(a.s6,b.s6,res.s5);
    res.s5 = mad24(a.s7,b.s5,res.s5);

    res.s4 = mad24(a.s4,b.s7,res.s5>>15);
    res.s4 = mad24(a.s5,b.s6,res.s4);
    res.s4 = mad24(a.s6,b.s5,res.s4);
    res.s4 = mad24(a.s7,b.s4,res.s4);

    res.s3 = mad24(a.s3&15,b.s7,res.s4>>15);
    res.s3 = mad24(a.s4,b.s6,res.s3);
    res.s3 = mad24(a.s5,b.s5,res.s3);
    res.s3 = mad24(a.s6,b.s4,res.s3);
    res.s3 = mad24(a.s7,b.s3,res.s3);

    res.s4567 = res.s4567&32767;
    res.s3 = res.s3&15;
    return res;
}

uint8 montmul(uint8 abar, uint8 bbar, uint8 m, uint8 mprime) {
    uint ttop;
    uint8 tmlo;

    uint8 t = {0,0,0,0,0,0,0,0};

    t = mul24_120(abar,bbar,t);
    ttop = mad24(abar.s3,bbar.s3,t.s0>>15);
    t = t&32767;
    
    /* Now compute u = (t + ((t*mprime) & mask)*m) >> 64. The mask is fixed at 2**64-1. Because it is a 64-bit quantity, it suffices to compute the low-order 64 bits of t*mprime, which means we can ignore thi. */
 
    tmlo = mul24_64(t,mprime);
    t = mul24_120(tmlo,m,t);
    //Work out the top 8 bits of t
    ttop = ttop + mad24(tmlo.s3,m.s3,t.s0>>15);
    t = t&32767;

    //The low half of t is contained in t.s4567 and the lowest 4 bits of t.s3
    //The high half of t is contained in the highest 11 bits of t.s3, t.s012 and ttop (8 bit).
    t.s7 = (t.s3 >> 4) + (t.s2 << 11);
    t.s6 = (t.s2 >> 4) + (t.s1 << 11);
    t.s5 = (t.s1 >> 4) + (t.s0 << 11);
    t.s4 = (t.s0 >> 4) + (ttop << 11);
    t.s3 = ttop;
    t.s2 = 0;
    t.s1 = 0;
    t.s0 = 0;
    t.s4567 = t.s4567&32767;
 
    return t;
}

uint8 ulong64_15 (ulong a) {
    uint8 m = (uint8) {0,0,0,a>>60,a>>45,a>>30,a>>15,a};
    m = m&32767;
    return m;
} 

long binExtEuclid(long a, long b){
   long u = b;
   long v = a;
   long r = 0;
   long s = 1;
   long x = 0;
   while (v>0){
      
      if ((u & 1)==0){
         u = u >> 1;
         if ((r % 2)==0){
            r = r >> 1;
         } else {
            r = (r + b) >> 1;
         }
      } else {
         if ((v & 1)==0){
            v = v >> 1;
            if ((s % 2)==0){
               s = s >> 1;
            } else {
               s = (s + b) >> 1;
            }
         } else {
            x = u - v;
            if (x>0){
               u = x;
               r = r - s;
               if (r<0){
                  r = r + b;
               } 
            } else {
               v = x * -1;
               s = s - r;
               if (s<0){
                  s = s + b;
               }
            }
         }
      }
   }
   if (r>=b){
      r = r - b;
   }
   if (r<0){
      r = r + b;
   }
   return(r);
}


__kernel void sieveKernel(
    __global long *KernelP,
    __global long *NOut,
    __global int *kns,
    __constant int *ks,
    int KernelBase,
    int NMax,
    int NMin,
    int counter,
    int numKs,
    int loop)

{
    int gid = get_global_id(0);
    ulong b = KernelP[gid];
    ulong2 rInvMdash = xbinGCD(0x8000000000000000UL, b);
        
    uint8 m = ulong64_15(b);
    uint8 r = ulong64_15(rInvMdash.y);

    long output = -5;
    int d = NMax-NMin;

    //Move b1 to montgomery space
    ulong bt = modul64(KernelBase,0,b);
    uint8 b1 = ulong64_15(bt);

    //These are all montgomery mults
    uint8 b2 = montmul(b1,b1,m,r);
    uint8 b3 = montmul(b2,b2,m,r);
    uint8 b4 = montmul(b3,b3,m,r);
    uint8 b5 = montmul(b4,b4,m,r);
    uint8 b6 = montmul(b5,b5,m,r);
    uint8 b7 = montmul(b6,b6,m,r);

    //Move the x0 starting point to montgomery space
    ulong x0t = modul64(1,0,b);
    uint8 x0 = ulong64_15(x0t);

    int tempNMax = NMax;

    uint8 bInc = b1;
    int k = 32 - clz(tempNMax);
    for (int i=0; i<k; i++) {
        if ((tempNMax&1) == 1) {
            x0 = montmul(x0,bInc,m,r);
        }
        tempNMax>>=1;
        bInc = montmul(bInc,bInc,m,r);
    }

    uint j = 0;
    uint8 bs[] = {b1,b3,b5,b7};

    for (int i=0; i<loop; i++){
        j = ((x0.s7)&3);
        d = d + (1<<j<<j);
        x0 = montmul(x0,bs[j],m,r);
    }

//The loop above is only run once as it doesn't depend on c1, and the loop below is run for each c1 (each k value)

    int permD = d;
    ulong c1t = 0;
    uint8 c1;
    //int count = 0;

    for (int i=0; i<1; i++) {
        d = permD;
        bool xor = 1;
        output = -5;
        c1t = binExtEuclid(ks[i],b);
        //Move this to montgomery space
        c1t = modul64(c1t,0,b);
        c1 = ulong64_15(c1t);

        while(xor) {
            //count++;
            j = ((c1.s7)&3);
            d = d - (1<<j<<j);
            c1 = montmul(c1,bs[j],m,r);

            if(c1.s7==x0.s7) {
                if (c1.s6==x0.s6 && c1.s5==x0.s5 && c1.s4==x0.s4 && c1.s3==x0.s3) {
                    output = d+NMin;
                    xor = 0;
                }
            }
//            xor = (c1.w!=x0.w) | (c1.z!=x0.z) | (c1.y!=x0.y);
//            output = (1-xor)*(d+NMin);
            xor = xor&(!(d>>31));        
            
        }

        if (output < NMin || output > NMax) {
            output=-3;
        }
        
        else {
            //We've had a match, check the n values
            int thek=ks[i];
            for (int y=0; y<counter; y++) {
                if (kns[y]==0) {
                    if(kns[y+1]==thek) {
                        //We are in the right k value, now check n values
                        for (int z = 2; z<counter; z++) {
                            if (kns[y+z]==0) {
                                //We've checked all the values for this k and not found a match
                                output=-2;
                                z = counter;
                                y=counter;
                            }
                            else if (kns[y+z]==output) {
                                z = counter;
                                y=counter;
                                i=numKs;
                                ulong t = thek;
                                t=t<<32;
                                output = t + output;
                            }
                        }
                    }
                }
            }
        }
    }

    NOut[gid] = output;
    
    return;
}