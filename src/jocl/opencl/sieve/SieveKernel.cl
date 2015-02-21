#pragma OPENCL EXTENSION cl_khr_fp64 : enable

int binlog(long x) {
    int log = 0;
    if(x>=4294967296L) {x >>= 32; log = 32;}
    if(x>=65536) {x >>= 16; log += 16;}
    if(x >= 256) {x >>= 8; log += 8;}
    if(x >= 16) {x >>= 4; log += 4;}
    if(x >= 4) {x >>= 2; log += 2;}
    x>>=1;
    log += (int)x;
    return log;
}

long mulmod(long x, long y, long m){
    int nonzeros = 0;
    if (x>y) {
        x^=y;
        y^=x;
        x^=y;
    }
    long res = 0;
    while (x > 0) { 
        nonzeros = binlog(x)+binlog(y);
        if (nonzeros < 62) {
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
    __global int *NOut,
    __global int *kn,
    int knlength,
    int KforKernel,
    int KernelBase,
    int NMax,
    int NMin)

{
    int gid = get_global_id(0);
    long b = KernelP[gid];
    long plessone = b-1;
    int plessoneint = 2147483647;
    if (plessone<2147483647) {
        plessoneint = (int)plessone;
    }

    int output = -5;
    long c1 = 0;
    int NRange = NMax-NMin;
    int shift = binlog(NRange)/2;
    int loops = 2<<(shift+1);
    //int loops = 100;
    c1 = binExtEuclid(KforKernel,b);
    long b1 = (long)KernelBase;
    long b2 = (b1 * b1) % b;
    long b3 = (b2 * b2) % b;
    long b4 = mulmod(b3, b3, b);
    long b5 = mulmod(b4, b4, b);
    long b6 = mulmod(b5, b5, b);
    long b7 = mulmod(b6, b6, b);
 
    long x0 = 1;
    int tempNMax = NMax;

    long bInc = KernelBase;
    int k = 1 + binlog(tempNMax);
    for (int i=0; i<k; i++) {
        if ((tempNMax&1) == 1) {
            x0 = mulmod(x0,bInc,b);
        }
        tempNMax>>=1;
        bInc = mulmod(bInc,bInc,b);
    }
    
    int d = NRange;
    int j = 0;
    int j1=0;
    int co1 = 0;
    int co2 = 0;
    int co3 = 0;
    int co4 = 0;
    long equation = 0;
    for (int i = 0; i<loops; i++){
        j = (((int)x0)&3);
        d = d + (1<<j<<j);
        j1 = (j&1);
        j = (j>>1);
        co1 = (1-j)&(1-j1);
        co2 = j1&(1-j);
        co3 = j&(1-j1);
        co4 = j1&j;
        equation = co1*b1 + co2*b3 + co3*b5 + co4*b7;
        
        x0 = mulmod(x0,equation,b);
    }

    bool l = true;
    while(l) {
        if (c1!=x0) {
            j = (((int)c1)&3);
            d = d - (1<<j<<j);
            j1 = (j&1);
            j = (j>>1);
            co1 = (1-j)&(1-j1);
            co2 = j1&(1-j);
            co3 = j&(1-j1);
            co4 = j1&j;
            equation = co1*b1 + co2*b3 + co3*b5 + co4*b7;
            c1 = mulmod(c1,equation,b);
                
            if (d < 0) {
                output = -1;
                l = false;
            }
        }

        else {
            output = (d+NMin)%plessoneint;
            while (output<NMin) {
                output = output + plessoneint;
            }
            l = false;
        }
    }

    if (output > NMax) {
        output = -3;
    }
    if (output<=NMax && output>=NMin) {
        int tempOut = -2;
        for (int s=0; s<knlength; s++) {
            if (output == kn[s]) {
                tempOut = output;
                s = knlength;
            }
        }       
        output = tempOut;
        if (output != -2) {
            x0 = 1;
            int temp = output;
            bInc = KernelBase;
            k = 1+ binlog(temp);
            for (int i=0; i<k; i++) {
                if ((temp&1) == 1) {
                    x0 = mulmod(x0,bInc,b);
                }
                temp>>=1;
                bInc = mulmod(bInc,bInc,b);
            }
            long rem = mulmod(KforKernel,x0,b);
            if (rem < 0) {
                rem = rem+b;
            }
            if (rem != 1) {
                output = -9;
            }
        }
    }
        
    NOut[gid] = output;
    
    return;
}


    