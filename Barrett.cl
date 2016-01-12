#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef struct _int128_t{ulong d1,d0;}int128_t;

int128_t div_128_64(ulong b) {
    /* Divides 2^128 by b, for 64-bit integer b, giving the quotient as the result. */
    int128_t quotient;

    ulong upper=1;

    for (int i=0; i<65; i++) {
        quotient.d1 = quotient.d1<<1;
        if (b<=upper) {
            upper = upper-b;
            quotient.d1++;
        }
        upper = upper<<1;
    }
        
    for (int i=0; i<64; i++) { 
        quotient.d0 = quotient.d0<<1;
        if (b<=upper) {
            upper = upper-b;
            quotient.d0++;
        }
        upper = upper<<1;
    }
   
    return quotient;
}

ulong barrett(int128_t m, ulong a, ulong b, ulong prime) {
//We currently do 6*64-bit multiplies in here. We can probably reduce this to increase speed. 
    ulong q;
    int128_t ab;
    //Calculate q = m.a.b/2^128. I.e. we only need the top 64 bits of m.a.b as m.a.b is at most 192 bits
    ab.d0 = a*b;
    ab.d1 = mul_hi(a,b);

    q = (ab.d1*m.d1) + mul_hi(ab.d1,m.d0) + mul_hi(ab.d0,m.d1);

    ulong r;
    //Calculate r = (a.b)-(q.n). This must be less than n so we only need the low 64 bits of (a.b) and (q.n)
    r = ab.d0 - (q*prime);

    if (r>prime) {
        r=r-prime;
        }

    return r;
}


long binExtEuclid(long a, long b){
   long u = b;
   long v = a;
   long r = 0;
   long s = 1;
   long x = a;
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

    long output = -5;
    int d = NMax;

    //Calculate m = floor(2^128/b) using div_128_64. 
    int128_t m;
    m = div_128_64(b);

    ulong b1 = KernelBase;
    ulong b2 = barrett(m,b1,b1,b);
    ulong b3 = barrett(m,b2,b2,b);
    ulong b4 = barrett(m,b3,b3,b);
    ulong b5 = barrett(m,b4,b4,b);
    ulong b6 = barrett(m,b5,b5,b);
    ulong b7 = barrett(m,b6,b6,b);

    ulong x0 = 1;
    int tempNMax = NMax;

    ulong bInc = KernelBase;
    int k = 32 - clz(tempNMax);
    for (int i=0; i<k; i++) {
        if ((tempNMax&1) == 1) {
            x0 = barrett(m,x0,bInc,b);
        }
        tempNMax>>=1;
        bInc = barrett(m,bInc,bInc,b);
    }

    int j = 0;
    int j1=0;
    ulong bsnew = 0;

    for (int i = 0; i<loop; i++){
        j = ((x0)&3);
        j1 = j&1;
        bsnew = j1==0 ? j==0 ? b1 : b5 : j==1 ? b3 : b7;
        d = d + (1<<j<<j);
        x0 = barrett(m,x0,bsnew,b);        
    }

//The loop above is only run once as it doesn't depend on c1, and the loop below is run for each c1 (each k value)

    int permD = d;
    ulong c1 = 0;

    for (int i=0; i<numKs; i++) {
        d = permD;
        bool xor = 1;
        output = -5;
        c1 = binExtEuclid(ks[i],b);
        while(xor) {
            j = ((c1)&3);
            j1 = j&1;
            bsnew = j1==0 ? j==0 ? b1 : b5 : j==1 ? b3 : b7;
            d = d - (1<<j<<j);
            c1 = barrett(m,c1,bsnew,b);

            xor = (c1!=x0);
            output = d;
            if (d<NMin) {
                xor=0;
            }
            
        }
        if (output < NMin) {
            output=-3;
        }
        else if (output > NMax) {
            output=-4;
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