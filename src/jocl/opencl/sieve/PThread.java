package jocl.opencl.sieve;

/**
 *
 * @author Rob
 */
public class PThread implements Runnable{

    int PSize;

    public PThread(int ArraySize) {
        PSize = ArraySize;
    }

    public void run() {
        long[] NewP = new long[PSize];
        NewP = JOCLOpenCLSieve.genPArray();
        JOCLOpenCLSieve.setPArray(NewP);
    }

}
