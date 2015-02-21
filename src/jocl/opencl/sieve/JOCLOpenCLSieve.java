package jocl.opencl.sieve;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import static org.jocl.CL.*;
import org.jocl.*;
import java.nio.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;

/**
 *
 * @author Rob Powell (rob147147)
 */
public class JOCLOpenCLSieve {
    
    private static cl_context context;
    private static cl_command_queue commandQueue;
    private static cl_kernel kernel;
    private static cl_program cpProgram;
    
    //Used for timing purposes
    static long startTime = 0;
    static long endTime = 0;

    static long KernelStartTime = 0;
    static long KernelWriteStartTime = 0;
    static long KernelWriteEndTime = 0;
    static long KernelExeStartTime = 0;
    static long KernelExeEndTime = 0;
    static long KernelReadStartTime = 0;
    static long KernelReadEndTime = 0;
    static long KernelEndTime = 0;
    static long KernelTotal = 0;

    static long FStartTime = 0;
    static long FEndTime = 0;
    static long FTotal = 0;

    //Set by ABCD file
    static int numKs = 0;
    static int[] base = new int[1];
    static ArrayList<ArrayList<Integer>> knpairs = new ArrayList();
    //Fixed by ABCD file base
    static double logkb = Math.log(base[0]);

    //Set by User on command line
    static int Pmin = 3;
    static long Pmax = 100000000;
    
    static String fileStr = "sr_108.abcd"; 

    //Used by program to traverse the range of P values set by user
    static long Pcurrent = Pmin;
    static int PArraySize = 16384*16;
    static long PEndOfLoop = 0;

    static int factors = 0;

    static int loopcounter = 0;

    static long[] KernelP;
    static long[] NextKernelP;
    
    //Used to control threads
    static final Semaphore usingPArray = new Semaphore(1, true);


    public static void main(String[] args) {
        printOpenCLInfo();
        initCL();
        
        String scalar = "1";
        int scale = Integer.parseInt(scalar);
        PArraySize = PArraySize*scale;

        //Read in ABCD file
        try {
            openABCD();
        }
        catch (IOException ioe) {
            System.out.println(ioe);
        }
        //Convert the knpairs ArrayList into an array.
        int size = 0;
        for (int i=0; i< knpairs.size(); i++) {
            size = size + knpairs.get(i).size();
            //Add extra space for delimiter
            size++;
        }
        
        int nMinint = Integer.MAX_VALUE;
        int nMaxint = 0;
        int counter=1;
        int counter1=0;
        //maxkn gives us the size of the longest row
        int[] kns = new int[size+1];
        kns[0] = 0;
        int[] ks = new int[knpairs.size()];
        //Put the entries of knpairs into kns
        for (int i=0; i<knpairs.size(); i++) {
            for (int j=0; j<knpairs.get(i).size(); j++) {
                kns[counter] = knpairs.get(i).get(j);
                if (j>0) {
                    if (kns[counter]<nMinint) {
                        nMinint = kns[counter];
                    }
                    if (kns[counter]>nMaxint) {
                        nMaxint = kns[counter];
                    }
                }
                else {
                    ks[counter1]= knpairs.get(i).get(j);
                    counter1++;
                }
                counter++;
            }
            //Insert delimiter at the end of the n-values for a given k-value
            kns[counter]=0;
            counter++;
        }
        //Printout to check we've added right
        //for (int i=0; i<kns.length; i++) {
        //    System.out.println(kns[i]);
        //}
        double diff = nMaxint-nMinint; 
        int loops = (int)Math.sqrt(diff);
        loops++;
        loops=loops<<1;
        //loops=(int)(loops*2);
        System.out.println("Loops: " + loops);
        int[] nMin = new int[1];
        int[] nMax = new int[1];
        int[] loop = new int[1];
        nMin[0] = nMinint;
        nMax[0] = nMaxint;
        loop[0] = loops;
        System.out.println("nMin: " + nMin[0]);
        System.out.println("nMax: " + nMax[0]);
        
        int[] numKs = new int[1];
        int[] count = new int[1];
        numKs[0] = knpairs.size();
        count[0] = counter;
        //End of reading ABCD file

        long[] NOut = new long[PArraySize];
        
        // Set the work-item dimensions
        long global_work_size[] = new long[]{PArraySize};
        
        //Allocate memory objects for input and output data
        cl_mem memObjects[] = new cl_mem[9];
        
        Pointer dst = Pointer.to(NOut);
        Pointer srcB = Pointer.to(kns);
        Pointer srcC = Pointer.to(ks);
        Pointer srcD = Pointer.to(base);
        Pointer srcE = Pointer.to(nMax);
        Pointer srcF = Pointer.to(nMin);
        Pointer srcG = Pointer.to(count);
        Pointer srcH = Pointer.to(numKs);
        Pointer srcI = Pointer.to(loop);
        
        //__global int *NOut
        memObjects[1] = clCreateBuffer(context, CL_MEM_READ_WRITE, Sizeof.cl_long * PArraySize, null, null);
        //__global int *kn
        memObjects[2] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * kns.length, srcB, null);
        //__global int *kn
        memObjects[3] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int * ks.length, srcC, null);
        //int KernelBase
        memObjects[4] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, srcD, null);
        //int NMax
        memObjects[5] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, srcE, null);
        //int NMin
        memObjects[6] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, srcF, null);
        //int count        
        memObjects[7] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, srcG, null);
        //int numKs
        memObjects[8] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, srcH, null);
        //int loop
        memObjects[8] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, srcI, null);
        
        
        clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(memObjects[1]));
        clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(memObjects[2]));
        clSetKernelArg(kernel, 3, Sizeof.cl_mem, Pointer.to(memObjects[3]));
        clSetKernelArg(kernel, 4, Sizeof.cl_int, srcD);
        clSetKernelArg(kernel, 5, Sizeof.cl_int, srcE);
        clSetKernelArg(kernel, 6, Sizeof.cl_int, srcF);
        clSetKernelArg(kernel, 7, Sizeof.cl_int, srcG);
        clSetKernelArg(kernel, 8, Sizeof.cl_int, srcH);
        clSetKernelArg(kernel, 9, Sizeof.cl_int, srcI);

        //Create a thread pool with just 1 thread. This will ensure we never use more than a single core for sieving
        ExecutorService executor = Executors.newFixedThreadPool(1);
        //We can use this 1 thread for pre-generating the next PArray and for searching/printing factors

        //Generate the first PArray
        KernelP = new long[PArraySize];
        NextKernelP = genPArray();
        Runnable newPGenerator = new PThread(PArraySize);
        
        
        startTime = System.currentTimeMillis();

        //Loop in here to go through multiple batches of P
        while (Pcurrent<Pmax) {
            
            //Acquire semaphore to update KernelP values for next GPU execution. If semaphore is unavailable the thread creating new P values is yet to finish
            try {
                usingPArray.acquire();
                KernelP = NextKernelP;
                System.out.println("First n value in this array is: " + KernelP[0]);
                System.out.println("Last n value in this array is: " + KernelP[KernelP.length-1]);
                usingPArray.release();
            }
            catch (Exception ex) {
                System.out.println("Exception occured");
            }
            //Acquire semaphore to generate the new P values for the GPU execution after this one
            try {
                usingPArray.acquire();
                executor.execute(newPGenerator);
            }
            catch (Exception ex) {
                System.out.println("Exception occured");
            }

            //Setup Pointers
            Pointer srcA = Pointer.to(KernelP);

            // __global int *KernelP
            memObjects[0] = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,Sizeof.cl_long * PArraySize, srcA, null);    

            //Time the kernel starts to run
            KernelStartTime = System.currentTimeMillis();
            KernelWriteStartTime = System.currentTimeMillis();

            //Send KernelP to the kernel
            clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(memObjects[0]));



            KernelWriteEndTime = System.currentTimeMillis();

            //Run the kernel
            KernelExeStartTime = System.currentTimeMillis();
            clEnqueueNDRangeKernel(commandQueue, kernel, 1, null, global_work_size, null, 0, null, null);                
            clFinish(commandQueue);
            KernelExeEndTime = System.currentTimeMillis();
            // Read the output data
            KernelReadStartTime = System.currentTimeMillis();
            clEnqueueReadBuffer(commandQueue, memObjects[1], CL_TRUE, 0, Sizeof.cl_long * PArraySize, dst, 0, null, null);
            KernelReadEndTime = System.currentTimeMillis();
            clReleaseMemObject(memObjects[0]);

            //Time the kernel completes execution
            KernelEndTime = System.currentTimeMillis();
            loopcounter++;
            //System.out.println("Kernel Write Time: " + (KernelWriteEndTime-KernelWriteStartTime) + "ms");
            //System.out.println("Kernel Execute Time: " + (KernelExeEndTime-KernelExeStartTime) + "ms");
            //System.out.println("Kernel Read Time: " + (KernelReadEndTime-KernelReadStartTime) + "ms");
            System.out.println("Total Kernel W+E+R Time: " + (KernelEndTime-KernelStartTime) + "ms");
            KernelTotal = KernelTotal + (KernelEndTime-KernelStartTime);

            FStartTime = System.currentTimeMillis();
            int kernelFactors=0;
            for (int d=0; d<PArraySize; d++) {
                if (NOut[d]>0) {
                    //Print the factor out
                    factors++;
                    kernelFactors++;
                    long temp = NOut[d];
                    temp = temp<<32;
                    temp = temp>>32;
                    addFactors(KernelP[d] + "|" + (NOut[d]>>32) + "*" + base[0] + "^" + temp + "-1\n");
                    //We could remove the N value from the array to speed up future searching
                }
            }
            //for (int d=0; d<PArraySize; d++) {
                //System.out.println("The output for " + KernelP[d] + " is: " + NOut[d]);
            //}
            FEndTime = System.currentTimeMillis();
            //System.out.println("Factor Execute Time: " + (FEndTime-FStartTime) + "ms");
            FTotal = FTotal + (FEndTime-FStartTime);

            //System.out.println("From GPU: " + NOut[0] + "," + NOut[1] + "," + NOut[2] + "," + NOut[3] + "," + NOut[4] + "," + NOut[5] + "," + NOut[6] + "," + NOut[7] + "," + NOut[8] + "," + NOut[9]);
            //System.out.println("Factors: " + KernelP[0] + "," + KernelP[1] + "," + KernelP[2] + "," + KernelP[3] + "," + KernelP[4] + "," + KernelP[5] + "," + KernelP[6] + "," + KernelP[7] + "," + KernelP[8] + "," + KernelP[9]);
            System.out.println("Factors in this kernel: " + kernelFactors);
            
         }

        executor.shutdown();
        endTime = System.currentTimeMillis();
        System.out.println(KernelTotal + "ms to complete the kernels");
        System.out.println(loopcounter + " kernel executions");
        System.out.println(FTotal + "ms to complete the factors code");
        System.out.println("Factors found: " + factors);
        System.out.println((Pmax/KernelTotal)*1000 + " p/sec");
        System.out.println(((Pmax/loopcounter)/(KernelEndTime-KernelStartTime))*1000 + " p/sec last kernel avg");
        
        // Release kernel, program, and memory objects
//        clReleaseMemObject(memObjects[0]);
//        clReleaseMemObject(memObjects[1]);
//        clReleaseMemObject(memObjects[2]);
//        clReleaseMemObject(memObjects[3]);
//        clReleaseMemObject(memObjects[4]);
//        clReleaseMemObject(memObjects[5]);
//        clReleaseMemObject(memObjects[6]);
//        clReleaseMemObject(memObjects[7]);
        clReleaseKernel(kernel);
        clReleaseProgram(cpProgram);
        clReleaseCommandQueue(commandQueue);
        clReleaseContext(context);

    }
    
        private static void initCL() {
            
        final int platformIndex = 0;
        final long deviceType = CL_DEVICE_TYPE_ALL;
        final int deviceIndex = 0;
        
        CL.setExceptionsEnabled(true);
        
        // Obtain the number of platforms
        int numPlatformsArray[] = new int[1];
        clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        
        //Obtain a platform ID
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms];
        clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[platformIndex];
        
        // Initialize the context properties
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform);
        
        // Obtain the number of devices for the platform
        int numDevicesArray[] = new int[1];
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        
        // Obtain a device ID 
        cl_device_id devices[] = new cl_device_id[numDevices];
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null);
        cl_device_id device = devices[deviceIndex];
        
        // Create a context for the selected device
        context = clCreateContext(contextProperties, 1, new cl_device_id[]{device}, null, null, null);
        
        // Create a command-queue for the selected device
        commandQueue = clCreateCommandQueue(context, device, 0, null);

        // Program Setup
        //C:\\Users\\Rob\\Documents\\NetBeansProjects\\JOCL OpenCL Sieve\\
        String source = readFile("SieveKernel.cl");

        // Create the program
        cpProgram = clCreateProgramWithSource(context, 1, new String[]{ source }, null, null);
        
        // Build the program
        clBuildProgram(cpProgram, 0, null, "-Werror", null, null);

        // Create the kernel
        kernel = clCreateKernel(cpProgram, "sieveKernel", null);
        
        System.out.println("Kernel Build Complete");
        
        }
        
    public static void setPArray(long[] PArray) {
        NextKernelP = PArray;
        usingPArray.release();
    }

    public static long[] genPArray() {
        int limit = 64*PArraySize;
        long[] ps = new long[PArraySize];
        int[] smallPrimes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,
            211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,
            467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,
            761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,
            1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,
            1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,
            1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,
            1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999};
        int size = smallPrimes.length;
        boolean[] set = new boolean [limit];
        for (int i=0; i<limit; i++) {
            set[i] = true;
        }
        
        for (int i=0; i<size; i++) {
            //For every small prime find the first number in range such that num%smallPrime[i] == 0
            int j=0;
            int p = smallPrimes[i];
            while (true) {
                long rem = (Pcurrent+j)%p;
                if (rem == 0) {
                    break;
                }
                j++;
            }
            for (int k=j; k<limit; k=k+p) {
                set[k] = false;
            }
            if (Pcurrent+j <= smallPrimes[size-1]) {
                set[j] = true;
            }
        }
       
        int j=0;
        for (int i=0; i<limit; i++) {
            //Iterate through all of the set bits and fill the array ps
            if (set[i] == true) {
                //Add to next free slot in ps
                ps[j] = Pcurrent + i;
                j++;
            }
            if (ps[PArraySize-1] != 0) {
                //ps is full, break this loop and update Pcurrent
                Pcurrent = ps[PArraySize-1];
                //System.out.println(i);
                break;
            }
        } 
        
        PEndOfLoop = ps[PArraySize-1];
        //System.out.println(PEndOfLoop);
        Pcurrent = PEndOfLoop;
        return ps;
    }

        public static void openABCD() throws IOException{
            //C:\\Users\\Rob\\Documents\\NetBeansProjects\\JOCL OpenCL Sieve\\
        File file = new File(fileStr);
        BufferedReader bufRdr = new BufferedReader(new FileReader(file));
        String line = null;
        while((line = bufRdr.readLine()) != null) {
            String[] array = new String[9];
            StringTokenizer st = new StringTokenizer(line, " ");
            int i =0;
            while (st.hasMoreTokens()) {
                array[i] = st.nextToken();
                i++;
            }
            //If array[0] == "ABCD" then this defines a new k
            //Otherwise string[0] is a number
            if(array[0].equals("ABCD")) {
                ArrayList<Integer> kn = new ArrayList();
                knpairs.add(kn);
                numKs++;
                //array[1] defines the k and b values
                //k
                StringTokenizer st1 = new StringTokenizer(array[1], "*");
                String s = st1.nextToken();
                int k = Integer.parseInt(s);
                System.out.println("k: " + s);
                String s2 = st1.nextToken();
                //b
                StringTokenizer st2 = new StringTokenizer(s2, "^");
                String s3 = st2.nextToken();
                System.out.println("b: " + s3);
                int b = Integer.parseInt(s3);
                //array[2] defines the first n value
                String s4 = array[2];
                s4 = s4.substring(1,s4.length()-1);
                System.out.println("First n: " + s4);
                //We don't need any of the other values
                int n = Integer.parseInt(s4);

                knpairs.get(numKs-1).add(k);
                base[0] = b;
                knpairs.get(numKs-1).add(n);
            }
            else {
                String nextN = array[0];
                int add = Integer.parseInt(nextN);
                int next = add + knpairs.get(numKs-1).get(knpairs.get(numKs-1).size()-1);
                knpairs.get(numKs-1).add(next);
            }
        }

        bufRdr.close();
    }

        public static synchronized void addFactors(String factor) {
            try {
                BufferedWriter out = new BufferedWriter(new FileWriter("factors.txt", true));
                out.write(factor);
                out.close();
            }
            catch (IOException e) {
                System.out.println("Error in the addFactors method");
            }
        }
        
    
        private static String readFile(String fileName) {
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
            StringBuffer sb = new StringBuffer();
            String line = null;
            while (true)
            {
                line = br.readLine();
                if (line == null)
                {
                    break;
                }
                sb.append(line).append("\n");
            }
            return sb.toString();
        }
        catch (IOException e) {
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }
    
    public static void printOpenCLInfo() {
                 // Obtain the number of platforms
        int numPlatforms[] = new int[1];
        clGetPlatformIDs(0, null, numPlatforms);

        System.out.println("Number of platforms: "+numPlatforms[0]);

        // Obtain the platform IDs
        cl_platform_id platforms[] = new cl_platform_id[numPlatforms[0]];
        clGetPlatformIDs(platforms.length, platforms, null);

        // Collect all devices of all platforms
        List<cl_device_id> devices = new ArrayList<cl_device_id>();
        for (int i=0; i<platforms.length; i++)
        {
            String platformName = getString(platforms[i], CL_PLATFORM_NAME);

            // Obtain the number of devices for the current platform
            int numDevices[] = new int[1];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, null, numDevices);

            System.out.println("Number of devices in platform "+platformName+": "+numDevices[0]);

            cl_device_id devicesArray[] = new cl_device_id[numDevices[0]];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices[0], devicesArray, null);

            devices.addAll(Arrays.asList(devicesArray));
        }

        // Print the infos about all devices
        for (cl_device_id device : devices)
        {
            // CL_DEVICE_NAME
            String deviceName = getString(device, CL_DEVICE_NAME);
            System.out.println("--- Info for device "+deviceName+": ---");
            System.out.printf("CL_DEVICE_NAME: \t\t\t%s\n", deviceName);

            // CL_DEVICE_VENDOR
            String deviceVendor = getString(device, CL_DEVICE_VENDOR);
            System.out.printf("CL_DEVICE_VENDOR: \t\t\t%s\n", deviceVendor);

            // CL_DRIVER_VERSION
            String driverVersion = getString(device, CL_DRIVER_VERSION);
            System.out.printf("CL_DRIVER_VERSION: \t\t\t%s\n", driverVersion);

            // CL_DEVICE_TYPE
            long deviceType = getLong(device, CL_DEVICE_TYPE);
            if( (deviceType & CL_DEVICE_TYPE_CPU) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_CPU");
            if( (deviceType & CL_DEVICE_TYPE_GPU) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_GPU");
            if( (deviceType & CL_DEVICE_TYPE_ACCELERATOR) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_ACCELERATOR");
            if( (deviceType & CL_DEVICE_TYPE_DEFAULT) != 0)
                System.out.printf("CL_DEVICE_TYPE:\t\t\t\t%s\n", "CL_DEVICE_TYPE_DEFAULT");

            // CL_DEVICE_MAX_COMPUTE_UNITS
            int maxComputeUnits = getInt(device, CL_DEVICE_MAX_COMPUTE_UNITS);
            System.out.printf("CL_DEVICE_MAX_COMPUTE_UNITS:\t\t%d\n", maxComputeUnits);

            // CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS
            long maxWorkItemDimensions = getLong(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS);
            System.out.printf("CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS:\t%d\n", maxWorkItemDimensions);

            // CL_DEVICE_MAX_WORK_ITEM_SIZES
            long maxWorkItemSizes[] = getSizes(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
            System.out.printf("CL_DEVICE_MAX_WORK_ITEM_SIZES:\t\t%d / %d / %d \n",
                maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            // CL_DEVICE_MAX_WORK_GROUP_SIZE
            long maxWorkGroupSize = getSize(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
            System.out.printf("CL_DEVICE_MAX_WORK_GROUP_SIZE:\t\t%d\n", maxWorkGroupSize);

            // CL_DEVICE_MAX_CLOCK_FREQUENCY
            long maxClockFrequency = getLong(device, CL_DEVICE_MAX_CLOCK_FREQUENCY);
            System.out.printf("CL_DEVICE_MAX_CLOCK_FREQUENCY:\t\t%d MHz\n", maxClockFrequency);

            // CL_DEVICE_ADDRESS_BITS
            int addressBits = getInt(device, CL_DEVICE_ADDRESS_BITS);
            System.out.printf("CL_DEVICE_ADDRESS_BITS:\t\t\t%d\n", addressBits);

            // CL_DEVICE_MAX_MEM_ALLOC_SIZE
            long maxMemAllocSize = getLong(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
            System.out.printf("CL_DEVICE_MAX_MEM_ALLOC_SIZE:\t\t%d MByte\n", (int)(maxMemAllocSize / (1024 * 1024)));

            // CL_DEVICE_GLOBAL_MEM_SIZE
            long globalMemSize = getLong(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            System.out.printf("CL_DEVICE_GLOBAL_MEM_SIZE:\t\t%d MByte\n", (int)(globalMemSize / (1024 * 1024)));

            // CL_DEVICE_ERROR_CORRECTION_SUPPORT
            int errorCorrectionSupport = getInt(device, CL_DEVICE_ERROR_CORRECTION_SUPPORT);
            System.out.printf("CL_DEVICE_ERROR_CORRECTION_SUPPORT:\t%s\n", errorCorrectionSupport != 0 ? "yes" : "no");

            // CL_DEVICE_LOCAL_MEM_TYPE
            int localMemType = getInt(device, CL_DEVICE_LOCAL_MEM_TYPE);
            System.out.printf("CL_DEVICE_LOCAL_MEM_TYPE:\t\t%s\n", localMemType == 1 ? "local" : "global");

            // CL_DEVICE_LOCAL_MEM_SIZE
            long localMemSize = getLong(device, CL_DEVICE_LOCAL_MEM_SIZE);
            System.out.printf("CL_DEVICE_LOCAL_MEM_SIZE:\t\t%d KByte\n", (int)(localMemSize / 1024));

            // CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE
            long maxConstantBufferSize = getLong(device, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
            System.out.printf("CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE:\t%d KByte\n", (int)(maxConstantBufferSize / 1024));

            // CL_DEVICE_QUEUE_PROPERTIES
            long queueProperties = getLong(device, CL_DEVICE_QUEUE_PROPERTIES);
            if(( queueProperties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ) != 0)
                System.out.printf("CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE");
            if(( queueProperties & CL_QUEUE_PROFILING_ENABLE ) != 0)
                System.out.printf("CL_DEVICE_QUEUE_PROPERTIES:\t\t%s\n", "CL_QUEUE_PROFILING_ENABLE");

            // CL_DEVICE_IMAGE_SUPPORT
            int imageSupport = getInt(device, CL_DEVICE_IMAGE_SUPPORT);
            System.out.printf("CL_DEVICE_IMAGE_SUPPORT:\t\t%d\n", imageSupport);

            // CL_DEVICE_MAX_READ_IMAGE_ARGS
            int maxReadImageArgs = getInt(device, CL_DEVICE_MAX_READ_IMAGE_ARGS);
            System.out.printf("CL_DEVICE_MAX_READ_IMAGE_ARGS:\t\t%d\n", maxReadImageArgs);

            // CL_DEVICE_MAX_WRITE_IMAGE_ARGS
            int maxWriteImageArgs = getInt(device, CL_DEVICE_MAX_WRITE_IMAGE_ARGS);
            System.out.printf("CL_DEVICE_MAX_WRITE_IMAGE_ARGS:\t\t%d\n", maxWriteImageArgs);

            // CL_DEVICE_SINGLE_FP_CONFIG
            long singleFpConfig = getLong(device, CL_DEVICE_SINGLE_FP_CONFIG);
            System.out.printf("CL_DEVICE_SINGLE_FP_CONFIG:\t\t%s\n",
                stringFor_cl_device_fp_config(singleFpConfig));

            // CL_DEVICE_IMAGE2D_MAX_WIDTH
            long image2dMaxWidth = getSize(device, CL_DEVICE_IMAGE2D_MAX_WIDTH);
            System.out.printf("CL_DEVICE_2D_MAX_WIDTH\t\t\t%d\n", image2dMaxWidth);

            // CL_DEVICE_IMAGE2D_MAX_HEIGHT
            long image2dMaxHeight = getSize(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT);
            System.out.printf("CL_DEVICE_2D_MAX_HEIGHT\t\t\t%d\n", image2dMaxHeight);

            // CL_DEVICE_IMAGE3D_MAX_WIDTH
            long image3dMaxWidth = getSize(device, CL_DEVICE_IMAGE3D_MAX_WIDTH);
            System.out.printf("CL_DEVICE_3D_MAX_WIDTH\t\t\t%d\n", image3dMaxWidth);

            // CL_DEVICE_IMAGE3D_MAX_HEIGHT
            long image3dMaxHeight = getSize(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT);
            System.out.printf("CL_DEVICE_3D_MAX_HEIGHT\t\t\t%d\n", image3dMaxHeight);

            // CL_DEVICE_IMAGE3D_MAX_DEPTH
            long image3dMaxDepth = getSize(device, CL_DEVICE_IMAGE3D_MAX_DEPTH);
            System.out.printf("CL_DEVICE_3D_MAX_DEPTH\t\t\t%d\n", image3dMaxDepth);

            // CL_DEVICE_PREFERRED_VECTOR_WIDTH_<type>
            System.out.printf("CL_DEVICE_PREFERRED_VECTOR_WIDTH_<t>\t");
            int preferredVectorWidthChar = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR);
            int preferredVectorWidthShort = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT);
            int preferredVectorWidthInt = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT);
            int preferredVectorWidthLong = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG);
            int preferredVectorWidthFloat = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT);
            int preferredVectorWidthDouble = getInt(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE);
            System.out.printf("CHAR %d, SHORT %d, INT %d, LONG %d, FLOAT %d, DOUBLE %d\n\n\n",
                   preferredVectorWidthChar, preferredVectorWidthShort,
                   preferredVectorWidthInt, preferredVectorWidthLong,
                   preferredVectorWidthFloat, preferredVectorWidthDouble);
        }
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static int getInt(cl_device_id device, int paramName)
    {
        return getInts(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private static int[] getInts(cl_device_id device, int paramName, int numValues)
    {
        int values[] = new int[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_int * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static long getLong(cl_device_id device, int paramName)
    {
        return getLongs(device, paramName, 1)[0];
    }

    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    private static long[] getLongs(cl_device_id device, int paramName, int numValues)
    {
        long values[] = new long[numValues];
        clGetDeviceInfo(device, paramName, Sizeof.cl_long * numValues, Pointer.to(values), null);
        return values;
    }

    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static String getString(cl_device_id device, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetDeviceInfo(device, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetDeviceInfo(device, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }

    /**
     * Returns the value of the platform info parameter with the given name
     *
     * @param platform The platform
     * @param paramName The parameter name
     * @return The value
     */
    private static String getString(cl_platform_id platform, int paramName)
    {
        // Obtain the length of the string that will be queried
        long size[] = new long[1];
        clGetPlatformInfo(platform, paramName, 0, null, size);

        // Create a buffer of the appropriate size and fill it with the info
        byte buffer[] = new byte[(int)size[0]];
        clGetPlatformInfo(platform, paramName, buffer.length, Pointer.to(buffer), null);

        // Create a string from the buffer (excluding the trailing \0 byte)
        return new String(buffer, 0, buffer.length-1);
    }
    
    /**
     * Returns the value of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @return The value
     */
    private static long getSize(cl_device_id device, int paramName)
    {
        return getSizes(device, paramName, 1)[0];
    }
    
    /**
     * Returns the values of the device info parameter with the given name
     *
     * @param device The device
     * @param paramName The parameter name
     * @param numValues The number of values
     * @return The value
     */
    static long[] getSizes(cl_device_id device, int paramName, int numValues)
    {
        // The size of the returned data has to depend on 
        // the size of a size_t, which is handled here
        ByteBuffer buffer = ByteBuffer.allocate(
            numValues * Sizeof.size_t).order(ByteOrder.nativeOrder());
        clGetDeviceInfo(device, paramName, Sizeof.size_t * numValues, 
            Pointer.to(buffer), null);
        long values[] = new long[numValues];
        if (Sizeof.size_t == 4)
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getInt(i * Sizeof.size_t);
            }
        }
        else
        {
            for (int i=0; i<numValues; i++)
            {
                values[i] = buffer.getLong(i * Sizeof.size_t);
            }
        }
        return values;
    }
    
}
