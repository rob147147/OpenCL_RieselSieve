#pragma OPENCL EXTENSION cl_khr_fp64 : enable

long mulmod(long a, long b, long m){
   int x = 63 - (int)floor((log((double)a) / log(2.0)));
   int y = 63 - (int)floor((log((double)b) / log(2.0)));
   long res = 0L;
   if ((x + y)>64){
      res = (a * b) % m;
   } else {
      long temp = 0L;
      if ((a - b)>0){
         temp = a;
         a = b;
         b = temp;
      }
      for (; (a - 0L)>0; b = (b << 1) % m){
         res = (res + (b * (a & 1L))) % m;
         a = a >> 1;
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
    __global int *KernelP,
    __global int *NOut,
    __global int *kn,
    int knlength,
    int KforKernel,
    int KernelBase,
    int NMax,
    int NMin)

{
    int gid = get_global_id(0);
    long b = (long)KernelP[gid];
    long plessone = b-1;
    long output = -5;
    long c1 = -1;
    int loops = 100;
    c1 = binExtEuclid(KforKernel,b);
    long b1 = (long)KernelBase;
    long b2 = (b1 * b1) % b;
    long b3 = (b2 * b2) % b;
    long b4 = mulmod(b3, b3, b);
    long b5 = mulmod(b4, b4, b);
    long b6 = mulmod(b5, b5, b);
    long b7 = mulmod(b6, b6, b);
    long b8 = mulmod(b7, b7, b);
    long b9 = mulmod(b8, b8, b);
    long b10 = mulmod(b9, b9, b);
    long b11 = mulmod(b10, b10, b);
    long b12 = mulmod(b11, b11, b);
    long b13 = mulmod(b12, b12, b);
    long x0 = 1;
    int tempNMax = NMax;
      if (tempNMax>=4096){
         x0 = mulmod(x0, b13, b);
         tempNMax = tempNMax - 4096;
      }
      if (tempNMax>=2048){
         x0 = mulmod(x0, b12, b);
         tempNMax = tempNMax - 2048;
      }
      if (tempNMax>=1024){
         x0 = mulmod(x0, b11, b);
         tempNMax = tempNMax - 1024;
      }
      if (tempNMax>=512){
         x0 = mulmod(x0, b10, b);
         tempNMax = tempNMax - 512;
      }
      if (tempNMax>=256){
         x0 = mulmod(x0, b9, b);
         tempNMax = tempNMax - 256;
      }
      if (tempNMax>=128){
         x0 = mulmod(x0, b8, b);
         tempNMax = tempNMax - 128;
      }
      if (tempNMax>=64){
         x0 = mulmod(x0, b7, b);
         tempNMax = tempNMax - 64;
      }
      if (tempNMax>=32){
         x0 = mulmod(x0, b6, b);
         tempNMax = tempNMax - 32;
      }
      if (tempNMax>=16){
         x0 = mulmod(x0, b5, b);
         tempNMax = tempNMax - 16;
      }
      if (tempNMax>=8){
         x0 = mulmod(x0, b4, b);
         tempNMax = tempNMax - 8;
      }
      if (tempNMax>=4){
         x0 = mulmod(x0, b3, b);
         tempNMax = tempNMax - 4;
      }
      if (tempNMax>=2){
         x0 = mulmod(x0, b2, b);
         tempNMax = tempNMax - 2;
      }
      if (tempNMax>=1){
         x0 = mulmod(x0, b1, b);
         tempNMax = tempNMax - 1;
      }

      long x = x0;
      long d = 0;
      long j = 0;
      long co1 = 0;
      long co2 = 0;
      long co3 = 0;
      long co4 = 0;
      long equation = 0;
      for (int i = 0; i<loops; i++){
         j = x & 3;
         co1 = ((1 - j) * (j - 2) * (j - 3)) / 6;
         co2 = (j * (j - 2) * (j - 3)) / 2;
         co3 = (j * (1 - j) * (j - 3)) / 2;
         co4 = (j * (j - 1) * (j - 2)) / 6;
         equation = (co1 * b1) + (co2 * b3) + (co3 * b5) + (co4 * b7);
         x = mulmod(x, equation, b);
         equation = co1 + (co2 * 4) + (co3 * 16) + (co4 * 64);
         d = d + equation;
      }

      long y = c1;
      long e = 0;
      bool loop = true;
      while(loop){
         
         if (y!=x){
            j = y & 3;
            co1 = ((1 - j) * (j - 2) * (j - 3)) / 6;
            co2 = (j * (j - 2) * (j - 3)) / 2;
            co3 = (j * (1 - j) * (j - 3)) / 2;
            co4 = (j * (j - 1) * (j - 2)) / 6;
            equation = (co1 * b1) + (co2 * b3) + (co3 * b5) + (co4 * b7);
            y = mulmod(y, equation, b);
            equation = co1 + (co2 * 4) + (co3 * 16) + (co4 * 64);
            e = e + equation;
            if (e > d+(NMax-NMin)){
               output = -1;
               loop = false;
            }
         } 
            else {
            output = ((d + (long)NMax) - e) % plessone;
            if (output<0){
               output = output + plessone;
            }
            loop = false;
         }
      }

      if ((output>=0) && (output<(long)NMin)){
         output = output + plessone;
         while (output<(long)NMin) {
             output = output + plessone;
         }
      }
      if (output > (long)NMax){
         output = -3;
      }
      int out = (int)output;
      if ((output<=(long)NMax) && (output>=(long)NMin)){
         bool loop = true;
         int s = 0;
         int tempOut = 0;
         while(loop){
         if (output==kn[s]){
             loop = false;
         } else {
             tempOut = -2;
         }
         s++;
         if (s==(knlength - 1)){
          loop = false;
          output = tempOut;
         }
        }
        out = tempOut;
      }
      NOut[gid]  = out;
      return;
}


    