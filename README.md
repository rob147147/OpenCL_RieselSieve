# OpenCL_RieselSieve
A JOCL based OpenCL sieve for numbers of the form k*b^n-1

To run this code requires the following 3 files:
JOCLOpenCLSieve.java
PThread.java
SieveKernel.cl

The algorithm implemented in SieveKernel.cl uses Pollard's Kangaroo Method for solving the discrete log problem.  
