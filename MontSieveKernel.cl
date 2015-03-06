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

ulong montmul(ulong abar, ulong bbar, ulong m, ulong mprime) {
    ulong thi, tlo, tm, tmmhi, tmmlo, uhi, ulo, ov;

    thi = mul_hi(abar,bbar);
    tlo = abar*bbar;
    
    /* Now compute u = (t + ((t*mprime) & mask)*m) >> 64. The mask is fixed at 2**64-1. Because it is a 64-bit quantity, it suffices to compute the low-order 64 bits of t*mprime, which means we can ignore thi. */
 
    tm = tlo*mprime;
    tmmhi = mul_hi(tm,m);
    tmmlo = tm*m;

    ulo = tlo + tmmlo; // Add t to tmm
    uhi = thi + tmmhi; // (128-bit add).
 
    if (ulo < tlo) uhi = uhi + 1; // Allow for a carry.
    // The above addition can overflow. Detect that here.
    ov = (uhi < thi) | ((uhi == thi) & (ulo < tlo));
    ulo = uhi; // Shift u right
    uhi = 0; // 64 bit positions.
    if (ov > 0 || ulo >= m) // If u >= m,
        ulo = ulo - m; // subtract m from u.
 
    return ulo;
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
               } else {
                } 
            } else {
               v = x * -1;
               s = s - r;
               if (s<0){
                  s = s + b;
               } else {
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
    ulong a = 0x8000000000000000UL;
    ulong2 rInvMdash = xbinGCD(a, b);

    long output = -5;
    int d = NMax-NMin;
    int loops = loop;

    //Move b1 to montgomery space
    ulong b1 = modul64(KernelBase,0,b);
    //These are all montgomery mults
    ulong b2 = montmul(b1,b1,b,rInvMdash.y);
    ulong b3 = montmul(b2,b2,b,rInvMdash.y);
    ulong b4 = montmul(b3,b3,b,rInvMdash.y);
    ulong b5 = montmul(b4,b4,b,rInvMdash.y);
    ulong b6 = montmul(b5,b5,b,rInvMdash.y);
    ulong b7 = montmul(b6,b6,b,rInvMdash.y);

    //Move the x0 starting point to montgomery space
    ulong x0 = modul64(1,0,b);
    int tempNMax = NMax;

    ulong bInc = b1;
    int k = 32 - clz(tempNMax);
    for (int i=0; i<k; i++) {
        if ((tempNMax&1) == 1) {
            x0 = montmul(x0,bInc,b,rInvMdash.y);
        }
        tempNMax>>=1;
        bInc = montmul(bInc,bInc,b,rInvMdash.y);
    }

    int j = 0;
    ulong bs[] = {b1,b3,b5,b7};
    
    for (int i = 0; i<loops; i++){
        j = ((x0)&3);
        d = d + (1<<j<<j);
        x0 = montmul(x0,bs[j],b,rInvMdash.y);
    }

//The loop above is only run once as it doesn't depend on c1, and the loop below is run for each c1 (each k value)

    int permD = d;
    ulong c1 = 0;
    //int count = 0;

    for (int i=0; i<numKs; i++) {
        d = permD;
        bool xor = 1;
        output = -5;
        c1 = binExtEuclid(ks[i],b);
        //Move this to montgomery space
        c1 = modul64(c1,0,b);
        while(xor) {
            //count++;
            j = ((c1)&3);
            d = d - (1<<j<<j);
            c1 = montmul(c1,bs[j],b,rInvMdash.y);

            xor = (c1!=x0);
            output = (1-xor)*(d + NMin);

            if (d<0) {
                xor=0;
            }        
            
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
                                //We've got a match, leave output and breakout
                                z = counter;
                                y=counter;
                                i=numKs;
                                //Can put a check in here just to doublecheck this is a factor 
                                //Move this to montgomery space
                                ulong xtemp = modul64(1,0,b);
                                int temp = output;
                                //This is already in montgomery space
                                bInc = b1;
                                k = 32 - clz(temp);
                                for (int x=0; x<k; x++) {
                                    if ((temp&1) == 1) {
                                        xtemp = montmul(xtemp,bInc,b,rInvMdash.y);
                                    }
                                    temp>>=1;
                                    bInc = montmul(bInc,bInc,b,rInvMdash.y);
                                }
                                //Move thek to montgomery space
                                ulong montk = modul64(thek,0,b);
                                ulong rem = montmul(montk,xtemp,b,rInvMdash.y);
                                //Move rem back to regular space to check that it equals 1
                                ulong phi = mul_hi(rem,rInvMdash.x);
                                ulong plo = rem*(rInvMdash.x);
                                rem = modul64(phi, plo, b); 
                                if (rem != 1) {
                                    output = -9;
                                }
                                else {
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
    }
   
    NOut[gid] = output;
    
    return;
}