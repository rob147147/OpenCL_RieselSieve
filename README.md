# OpenCL_RieselSieve
A JOCL based OpenCL sieve for numbers of the form k*b^n-1

To run this code requires the following 3 files:
JOCLOpenCLSieve.java<br />
PThread.java<br />
MontSieveKernel.cl<br />

The algorithm implemented in MontSieveKernel.cl uses Pollard's Kangaroo Method for solving the discrete log problem and Montgomery multiplication.

Current Performance on ATI HD 4850 (1000 GFlops):
1 k with n range of 1,000: 1.75x quicker than i5-4440
1 k with n range of 100,000: 7.8x slower than i5-4440 :(
36 k with n range of 190,000: 45.5x slower than i5-4440 :(  
