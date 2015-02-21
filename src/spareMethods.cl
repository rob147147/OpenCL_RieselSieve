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

long mulmod2(long x, long y, long m){
    if (x>y) {
        x^=y;
        y^=x;
        x^=y;
    }
    long res = 0;
    while (x>0) {
        if (x&1){
            res=res+y;
            if (res>=m) {
                res=res-m;
            }
        }
        x >>= 1;
        y <<= 1;
        if (y>=m) {
            y=y-m;
        }
    }
   return(res);
}

long mulmod3(long x, long y, long m, float bInv){
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
            res = (res + (x*y));
            res=res-(m*(long)(floor(res*bInv)));
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


ulong2 mul64(ulong x, ulong y){
    ulong f, o, i, l;
    ulong high, low;
    ulong2 highlow;

    ulong x_hi = x >> 32;
    ulong x_lo = x & UINT_MAX;
    ulong y_hi = y >> 32;
    ulong y_lo = y & UINT_MAX;

    f = x_hi * y_hi;
    o = x_hi * y_lo;
    i = x_lo * y_hi;
    l = x_lo * y_lo;

    high = (f + (hadd(o, (i + (l>>32))) >> 31));
    low = (l + ((i + o)<<32));

    highlow.x = high;
    highlow.y = low;

    return highlow;
}


long mulmod2(ulong a, ulong b, ulong m) {
    ulong res = 0;
    ulong temp_b;

    if (a>b) {
        a^=b;
        b^=a;
        a^=b;
    }

    while (a != 0) {
        if (a & 1) {
            if (b >= m - res) /* Equiv to if (res + b >= m), without overflow */
                res -= m;
            res += b;
        }
        a >>= 1;

        /* Double b, modulo m */
        temp_b = b;
        if (b >= m - b)       /* Equiv to if (2 * b >= m), without overflow */
            temp_b -= m;
        b += temp_b;
    }
    return (long)res;
}

long mulmod(long x, long y, long m){
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