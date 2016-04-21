#pragma OPENCL EXTENSION cl_amd_printf : enable

typedef struct _int128_t{ulong d1,d0;}int128_t;

int128_t div_128_64(ulong b) {
    /* Divides 2^128 by b, for 64-bit integer b, giving the quotient as the result. */
    int128_t quotient;

    ulong upper=2;

    for (int i=0; i<64; i++) {
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
    int128_t ab;
    //Calculate q = m.a.b/2^128. I.e. we only need the top 64 bits of m.a.b as m.a.b is at most 192 bits
    ab.d0 = a*b;
    ab.d1 = mul_hi(a,b);

//First two terms here are 0 if a and b are 32 bits or less as ab.d1=0
    ulong q = (ab.d1*m.d1) + mul_hi(ab.d1,m.d0) + mul_hi(ab.d0,m.d1);

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

    //int count = 0;

    //Calculate m = floor(2^128/b) using div_128_64. 
    int128_t m1;
    m1 = div_128_64(b);
    
    int m = 1024;
    int shift = 10; //m=2^shift

    //For all j s.t 0<=j<m, calculate KernelBase^j and store
    ulong js[1024]; //Would like to call this m but it won't compile as the memory requirements must be known in advance
    js[0] = 1;
    for (int j=1; j<m; j++) {
        js[j] = barrett(m1,js[j-1],KernelBase,b);
        //count++;
    }

    //Compute KernelBase^-m (mod b)
    ulong c1 = binExtEuclid(KernelBase,b); //This should be KernelBase^-1 (mod b)
    //Now repeatedly square it as m is a power of two
    for (int s=0; s<shift; s++) {
        c1 = barrett(m1,c1,c1,b);
        //count++;
    }
   
//Lets try changing this section - rather than looking at every possible match lets just look for the ones we're interested in
//The structure of the candidate file is 0,k,n-values,0,k,n-values,...
//counter is the length of this array, so just work through it

    long output = -5;

    for (int k=0; k<counter; k++) {
        if (kns[k]==0) {
            //The next entry is a k-value
            k++;
            //So work out beta from it
            int kval = kns[k];
            ulong beta = binExtEuclid(kns[k],b);
            //The next value is the first n-value for this k-value
            k++;
            bool first = true;
            int t=0;
            for (int z=0; z<counter; z++) {
                //Work through the n-values until we come across a zero, which implies we are done for this k-value
                if (kns[k+z] == 0) {
                    k=k+z-1; //This sets us back one value, so when the next loop starts and adds 1 to the value of k then we'll be located at a zero
                    z=counter;
                }
                //Otherwise this is an n-value and we need to check it
                //Work out tMin -> take the n-value and divide by m
                int n = kns[k+z];
                if (first) {
                    int tMin = n>>shift;
                    for (t=0; t<tMin; t++) {
                        beta = barrett(m1,beta,c1,b);
                        //count++;
                    }
                    first = false;
                    t = t*m;
                }
                //Check the difference between t*m and the n-value;
                int diff = n-t;
                if (diff>m) { //Changing this if to while seems to result in odd runtime when changing the input arraysize when it should have no effect!
                    diff=diff-m;
                    t=t+m;
                    beta = barrett(m1,beta,c1,b);
                    //count++;
                }
                //count++;
                if ((beta) == js[diff]) {
                    output = kval;
                    output = output<<32;
                    output = output + t + diff;
                }
            }
        }
    }
   
    NOut[gid] = output; //This contains the k-value in the top 32 bits and the n-value in the low 32 bits
    
    return;
}