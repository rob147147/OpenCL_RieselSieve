# OpenCL_RieselSieve
A JOCL based OpenCL sieve for numbers of the form k*b^n-1

To run this code requires 3 files:<br />
JOCLOpenCLSieve.java<br />
PThread.java<br />
Any one of the .cl files - each of which provide a different implementation<br />

The algorithm implemented in MontSieveKernel.cl, MontMul24SieveKernel.cl and MontMul24_15SieveKernel.cl uses Pollard's Kangaroo Method for solving the discrete log problem and Montgomery Multiplication to avoid expensive modulo operations. This method is probabilistic and often needs tuning to obtain the best results.<br />
MontSieveKernel.cl uses 64-bit multiplication, MontMul24SieveKernel.cl uses 24-bit mul24 operations on 16-bit integers and MontMul24_15SieveKernel.cl uses 24-bit mul24/mad24 operations on 15-bit integers.

The algorithm implemented in Barrett.cl again uses Pollard's Kangaroo Method for solving the discrete log problem, but this time uses Barrett Multiplication to avoid expensive modulo operations. Again due to the use of the Kangaroo method the algorithm is probabilistic and often needs tuning to obtain the best results.

Finally in BSGS.cl I have implemented a version of the Baby Step Giant Step algorithm. This requires less tuning, but the memory operations tend to be quite expensive on GPUs. 


Current Performance on my R7 260X (1881.6 GFlops) to be updated...  
