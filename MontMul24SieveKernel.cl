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

uint8 mul24_128(uint4 a, uint4 b) {
    //Multiply two 64 bit ulongs using mul24. 
    //Convert a and b into int vectors where each int contains only 16 bits of useful information.
       
    //We want all 16 terms from multiplying a and b together
    
    //c1Vec = {aw,bw,cw,dw}, c2Vec = {ax,bx,cx,dx}, c3Vec = {ay,by,cy,dy}, c4Vec = {az,bz,cz,dz}
    uint4 c1Vec = mul24(a,b.x);
    uint4 c2Vec = mul24(a,b.y);
    uint4 c3Vec = mul24(a,b.z);
    uint4 c4Vec = mul24(a,b.w);

    uint8 res = {0,0,0,0,0,0,0,0};
    //Use a long to keep a track of any overflows
    ulong ov = 0;
    //This could be as much as 32 bits. Keep the low 16 bits and add the top 16 bits to the next int
    ov = c4Vec.w;
    res.s7 = (ov)&65535;
    ov=ov>>16;
    ov = ov + c4Vec.z + c3Vec.w;
    res.s6 = (ov)&65535;
    ov=ov>>16;
    ov = ov + c4Vec.y + c3Vec.z + c2Vec.w;
    res.s5 = (ov)&65535;
    ov=ov>>16;
    ov = ov + c4Vec.x + c3Vec.y + c2Vec.z + c1Vec.w;
    res.s4 = (ov)&65535;
    ov=ov>>16;
    ov = ov + c3Vec.x + c2Vec.y + c1Vec.x;
    res.s3 = (ov)&65535;
    ov=ov>>16;
    ov = ov + c2Vec.x + c1Vec.y;
    res.s2 = (ov)&65535;
    ov=ov>>16;
    ov = ov + c1Vec.x;
    res.s1 = (ov)&65535;
    ov=ov>>16;
    res.s0 = ov;

    return res;
}

uint4 mul24_64(uint4 a, uint4 b) {
    //Return the low half of multiplying two 64 bit ulongs using mul24. 
    
    //c1Vec = {aw,bw,cw,dw}, c2Vec = {ax,bx,cx,dx}, c3Vec = {ay,by,cy,dy}, c4Vec = {az,bz,cz,dz}
    uint dw = mul24(a.w,b.x);
    uint2 c2Vec = mul24(a.zw,b.y);
    uint4 c3Vec = mul24(a,b.z);
    uint4 c4Vec = mul24(a,b.w);

    uint4 res = {0,0,0,0};
    //Use a long to keep a track of any overflows
    ulong ov = 0;
    //This could be as much as 32 bits. Keep the low 16 bits and add the top 16 bits to the next int
    ov = c4Vec.w;
    res.s3 = ov;
    ov=ov>>16;
    ov = ov + c4Vec.z + c3Vec.w;
    res.s2 = ov;
    ov=ov>>16;
    ov = ov + c4Vec.y + c3Vec.z + c2Vec.y;
    res.s1 = ov;
    ov=ov>>16;
    ov = ov + c4Vec.x + c3Vec.y + c2Vec.x + dw;
    res.s0 = ov;
    res = res&65535;

    return res;
}

uint4 montmul(uint4 abar, uint4 bbar, uint4 m, uint4 mprime) {
    uint8 t,tmm;
    uint4 tmlo;

    t = mul24_128(abar,bbar);
    
    /* Now compute u = (t + ((t*mprime) & mask)*m) >> 64. The mask is fixed at 2**64-1. Because it is a 64-bit quantity, it suffices to compute the low-order 64 bits of t*mprime, which means we can ignore thi. */
 
    tmlo = mul24_64(t.s4567,mprime);
    tmm = mul24_128(tmlo,m);

    //We need to compute t+tmm.
    t = t+tmm;
    //Deal with the overflows that may have occurred from adding two 16 bit numbers together
    int ov = 0;
    ov = t.s7>>16;
    t.s6 = t.s6+ov;
    ov = t.s6>>16;
    t.s5 = t.s5+ov;
    ov = t.s5>>16;
    t.s4 = t.s4+ov;
    ov = t.s4>>16;
    t.s3 = t.s3+ov;
    ov = t.s3>>16;
    t.s2 = t.s2+ov;
    ov = t.s2>>16;
    t.s1 = t.s1+ov;
    ov = t.s1>>16;
    t.s0 = t.s0+ov;
    //ov = t.s0>>16;

    t.s0123 = t.s0123&65535;
    //If t.s0 has overflowed then we need to subtract m. We also need to subtract m if t.hi>m

    //Short term fix - convert top half and m back to longs and do the subtraction, then convert the answer back
    //Algorithm seems to work without worrying about subtracting m
    //ulong hi = t.s0;
    //hi = hi<<16;
    //hi = hi + t.s1;
    //hi = hi<<16;
    //hi = hi + t.s2;
    //hi = hi<<16;
    //hi = hi + t.s3;

    //ulong ml = m.s0;
    //ml = ml<<16;
    //ml = ml + m.s1;
    //ml = ml<<16;
    //ml = ml + m.s2;
    //ml = ml<<16;
    //ml = ml + m.s3;

    //if (hi>=ml || ov>0) {
    //    hi = hi-ml;
    // }

     //Convert hi back
    //ulong h1 = hi>>16;
    //ulong h2 = h1>>16;
    //ulong h3 = h2>>16;
    //uint4 h = (uint4) {h3,h2,h1,hi};
    //h = h&65535;

    //uint4 h = (uint4) {t.s0,t.s1,t.s2,t.s3};
 
    return t.s0123;
}

uint4 ulong64_16 (ulong a) {
    ulong m1 = a>>16;
    ulong m2 = m1>>16;
    ulong m3 = m2>>16;
    uint4 m = (uint4) {m3,m2,m1,a};
    m = m&65535;
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
        
    uint4 m = ulong64_16(b);
    uint4 r = ulong64_16(rInvMdash.y);

    long output = -5;
    int d = NMax-NMin;

    //Move b1 to montgomery space
    ulong bt = modul64(KernelBase,0,b);
    uint4 b1 = ulong64_16(bt);

    //These are all montgomery mults
    uint4 b2 = montmul(b1,b1,m,r);
    uint4 b3 = montmul(b2,b2,m,r);
    uint4 b4 = montmul(b3,b3,m,r);
    uint4 b5 = montmul(b4,b4,m,r);
    uint4 b6 = montmul(b5,b5,m,r);
    uint4 b7 = montmul(b6,b6,m,r);

    //Move the x0 starting point to montgomery space
    ulong x0t = modul64(1,0,b);
    uint4 x0 = ulong64_16(x0t);

    int tempNMax = NMax;

    uint4 bInc = b1;
    int k = 32 - clz(tempNMax);
    for (int i=0; i<k; i++) {
        if ((tempNMax&1) == 1) {
            x0 = montmul(x0,bInc,m,r);
        }
        tempNMax>>=1;
        bInc = montmul(bInc,bInc,m,r);
    }

    int j = 0;
    uint4 bs[] = {b1,b3,b5,b7};

    for (int i=0; i<loop; i++){
        j = ((x0.w)&3);
        d = d + (1<<j<<j);
        x0 = montmul(x0,bs[j],m,r);
    }

//The loop above is only run once as it doesn't depend on c1, and the loop below is run for each c1 (each k value)

    int permD = d;
    ulong c1t = 0;
    uint4 c1;
    //int count = 0;

    for (int i=0; i<numKs; i++) {
        d = permD;
        bool xor = 1;
        output = -5;
        c1t = binExtEuclid(ks[i],b);
        //Move this to montgomery space
        c1t = modul64(c1t,0,b);
        c1 = ulong64_16(c1t);

        while(xor) {
            //count++;
            j = ((c1.w)&3);
            d = d - (1<<j<<j);
            c1 = montmul(c1,bs[j],m,r);

            if(c1.w==x0.w) {
                if (c1.z==x0.z && c1.y==x0.y && c1.x==x0.x) {
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