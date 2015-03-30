# OpenCL_RieselSieve
A JOCL based OpenCL sieve for numbers of the form k*b^n-1

To run this code requires the following 3 files:<br />
JOCLOpenCLSieve.java<br />
PThread.java<br />
MontSieveKernel.cl or MontMul24SieveKernel.cl

The algorithm implemented in MontSieveKernel.cl (& MontMul24SieveKernel.cl) uses Pollard's Kangaroo Method for solving the discrete log problem and Montgomery multiplication to avoid expensive modulo operations. MontSieveKernel.cl uses 64-bit multiplication, whereas MontMul24SieveKernel.cl uses 24-bit mul24 operations which can be considerably quicker on modern AMD GPUs

Current Performance on ATI HD 4850 (1000 GFlops) with MontSieveKernel.cl:<br />
1 k with n range of 1,000: 1.75x quicker than i5-4440<br />
1 k with n range of 100,000: 7.8x slower than i5-4440 :(<br />
36 k with n range of 190,000: 45.5x slower than i5-4440 :(  
