long mulmod(ulong x, ulong y, ulong m){

    ulong a = mul_hi(x,y);
    ulong b = x*y;

    if (a==0) {
        return (b%m);
    }

    //uint2 xVec = {(uint)(x>32),(uint)(x)};
    //uint2 yVec = {(uint)(y>32),(uint)(y)};

//Shift a as far left as possible, modulo m
//Shift a as far left as possible again and modulo m
//Keep shifting left and modulo'ing until shifted left 64 times
//The remainder + b%m modulo m is the answer
  
    int zeros = 0;
    if (x>y) {
        x^=y;
        y^=x;
        x^=y;
    }
    long res = 0;
    while (x > 0) { 
        zeros = clz(x)+clz(y);
        if (zeros > 65) {
            res = (res + (x*y))%m;
            x=0;
        }
        else {
            res = (res + (y*(x&1)))%m;
            x >>= 1;
            y = (y << 1)%m;
        }
    }
   return(res);
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
    __global int *ks,
    int KernelBase,
    int NMax,
    int NMin,
    int counter,
    int numKs,
    int loop)

{
    int gid = get_global_id(0);
    ulong b = KernelP[gid];
    //int plessone = 0; 
    //if (b>INT_MAX) {
    //    plessone = INT_MAX;
   // }
    //else {
    //    plessone = (int)(b-1);
   // }

    int c1s[] = {0};

    long output = -5;
    ulong c1 = 0;
    int d = NMax-NMin;
    int loops = loop;
    for (int i=0; i<numKs; i++) {
        c1s[i] = binExtEuclid(ks[i],b);
    }
    ulong b1 = (ulong)KernelBase % b;
    ulong b2 = (b1 * b1) % b;
    ulong b3 = (b2 * b2) % b;
    ulong b4 = mulmod(b3, b3, b);
    ulong b5 = mulmod(b4, b4, b);
    ulong b6 = mulmod(b5, b5, b);
    ulong b7 = mulmod(b6, b6, b);
 
    ulong x0 = 1;
    int tempNMax = NMax;

    ulong bInc = KernelBase;
    int k = 32 - clz(tempNMax);
    for (int i=0; i<k; i++) {
        if ((tempNMax&1) == 1) {
            x0 = mulmod(x0,bInc,b);
        }
        tempNMax>>=1;
        bInc = mulmod(bInc,bInc,b);
    }
    
    int j = 0;
    ulong bs[] = {b1,b3,b5,b7};
    for (int i = 0; i<loops; i++){
        j = ((x0)&3);
        d = d + (1<<j<<j);
        x0 = mulmod(x0,bs[j],b);
    }

//The loop above is only run once as it doesn't depend on c1, and the loop below is run for each c1 (each k value)

    int permD = d;
    bool xor = 1;

    for (int i=0; i<numKs; i++) {
        d = permD;
        c1 = c1s[i];
        while(xor) {

            j = ((c1)&3);
            d = d - (1<<j<<j);
            c1 = mulmod(c1,bs[j],b);           

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
                                ulong xtemp = 1;
                                int temp = output;
                                bInc = KernelBase;
                                k = 32 - clz(temp);
                                for (int x=0; x<k; x++) {
                                    if ((temp&1) == 1) {
                                        xtemp = mulmod(xtemp,bInc,b);
                                    }
                                    temp>>=1;
                                    bInc = mulmod(bInc,bInc,b);
                                }
                                ulong rem = mulmod(thek,xtemp,b);
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
   
    NOut[gid] = x0;
    
    return;
}


    