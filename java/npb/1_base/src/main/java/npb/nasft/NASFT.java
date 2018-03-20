/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package npb.nasft;

import static npb.nasft.FTProblemClass.*;


public class NASFT {

    // data for kernel computation
    private double[][] twiddle;
    private double[][] u0, u1, u2;
    private double[][] cplxSums;
    private double[][] u;
    
    private int[][] xstart;
    private int[][] xend;
    private int[][] ystart;
    private int[][] yend;
    private int[][] zstart;
    private int[][] zend;

    private int fftblock;
    private int fftblockpad;

    private int[] dims1;

    private int checkSumCount;
    private Complex[][][] z;
    private double[][] chunks;
    private double[][][] cplxTemps;

    private int nx,ny,nz, np2;
    private int iterations, ntotal_f, maxdim;
    private int[][] dims;
    private int layout_type;
    private int dataSize;

    FTProblemClass FTProblem;
    long time;
    public static void main(String[] args)
        throws Exception{
        char problemSize = 'S';
        int np = 32;
        for (int i = 0; i < args.length; i++) {
            if (args[i].compareToIgnoreCase("-np") == 0) {
                np = Integer.parseInt(args[++i]);
            } else if (args[i].compareToIgnoreCase("-class") == 0) {
                if (args[++i].length() == 1) {
                    problemSize = args[i].toUpperCase().charAt(0);
                } else {
                    System.err.println("Parse args error: " + args[i - 1] + " must have a one char parameter");
                    System.exit(1);
                }
            }
        }
        new NASFT(problemSize, np);
    }

    NASFT(char problemSize, int numberProcs)
        throws Exception{
        boolean verified=true;



        FTProblem = ClassFactory.getFTNASClass(problemSize, numberProcs);
        nx=FTProblem.nx;
        ny=FTProblem.ny;
        nz=FTProblem.nz;
        ntotal_f=(int)FTProblem.ntotal_f;
        iterations =FTProblem.iterations;
        maxdim =FTProblem.maxdim;
        dims=FTProblem.dims;
        np2=FTProblem.np2;
        layout_type=FTProblem.layout_type;
        dataSize =FTProblem.ntdivnp / FTProblem.np;

        init(numberProcs);
        u1 = new double[numberProcs][];
        System.out.println("WARMING UP!");
        //Warm up
        for (int rank=0;rank<numberProcs;rank++){
            setup(FTProblem, rank);           
        }
        
        allocation(numberProcs);
        
        FFT(numberProcs, problemSize);

        //Waiting for all previous tasks
        for (int rank=0;rank<numberProcs;rank++){
        	NASFTImpl.synchronize(u1[rank]);
        }
        
        System.out.println("RUNNING!");
        time=System.currentTimeMillis();

        //COMPUTING
        FFT(numberProcs, problemSize);
        
        for (int iteration=0;iteration<iterations;iteration++){
            inverseFFT(numberProcs, iteration);
            
            reduction(numberProcs);
        }

        System.out.println("VERIFYING");
        verified = verify(FTProblem);
        time=System.currentTimeMillis()-time;
        
        printEnd(time/1000.0, getMflops(time), verified);
    }


    private boolean verify(FTProblemClass FTProblem) {
        int nt = FTProblem.niter;
        int i;

        for (i = 1; i <= nt; i++) {
        	if (NASFTImpl.verify(FTProblem, cplxSums[i], i) == false)
        		return false;
        }
        return true;
    }
    
    public static void printStarted(String kernel, char className, long[] size, int nbIteration, int nbProcess) {
        System.out.print("\n\n NAS Parallel Benchmarks ProActive -- " + kernel + " Benchmark\n\n");
        System.out.println(" Class: " + className);
        System.out.print(" Size:  " + size[0]);
        for (int i = 1; i < size.length; i++) {
            System.out.print(" x " + size[i]);
        }
        System.out.println();

        System.out.println(" Iterations:   " + nbIteration);
        System.out.println(" Number of processes:     " + nbProcess);
    }

    public void printEnd(double totalTime, double mops, boolean passed_verification) {
        String verif;
        String javaVersion = System.getProperty("java.vm.vendor") + " " + System.getProperty("java.vm.name")
                + " " + System.getProperty("java.vm.version") + " - Version " + System.getProperty("java.version");

        verif = passed_verification ? "SUCCESSFUL" : "UNSUCCESSFUL";

        System.out.println("\n\n " + FTProblem.kernelName + " Benchmark Completed");
        System.out.println(" Class            =  " + FTProblem.problemClassName);
        System.out.println(" Size             =  " + FTProblem.size);
        System.out.println(" Iterations       =  " + FTProblem.iterations);
        System.out.println(" Time in seconds  =  " + totalTime);
        System.out.println(" Total processes  =  " + FTProblem.numProcs);
        System.out.println(" Mop/s total      =  " + mops);
        System.out.println(" Mop/s/process    =  " + (mops / FTProblem.numProcs));
        System.out.println(" Operation type   =  " + FTProblem.operationType);
        System.out.println(" Verification     =  " + verif);
        System.out.println(" NPB Version      =  " + FTProblem.version);
        System.out.println(" Java RE          =  " + javaVersion);
    }

    private double getMflops(long total) {
        double ntf = FTProblem.size;
        double time = total;
        double mflops = ntf *
            (14.8157 + 7.19641 * Math.log(ntf) + (5.23518 + 7.21113 * Math.log(ntf)) * FTProblem.niter) / time /
            1000.0;
        return mflops;
    }

    private void setup(FTProblemClass FTProblem, int rank) {
        z [rank]= new Complex[NASFTImpl.transblockpad + 1][NASFTImpl.transblock + 1];
        for (int i = 0; i <= NASFTImpl.transblockpad; i++)
            for (int j = 0; j <= NASFTImpl.transblock; j++)
                z[rank][i][j] = new Complex();




        // Determine processor coordinates of this processor
        // Processor grid is np1xnp2.
        // Arrays are always (n1, n2/np1, n3/np2)
        // Processor coords are zero-based.
        // emulation of comm_split with plan topology
        int me2 = rank % FTProblem.np2; // goes from 0...np2-1
        int me1 = rank / FTProblem.np2; // goes from 0...np1-1


        // size + 1 for translation from FORTRAN
        xstart[rank] = new int[3];
        xend[rank] = new int[3];
        ystart[rank] = new int[3];
        yend[rank] = new int[3];
        zstart[rank] = new int[3];
        zend[rank] = new int[3];

        // Determine which section of the grid is owned by this processor
        if (FTProblem.layout_type == 0) {
            for (int i = 0; i < 3; i++) {
                xstart[rank][i] = 1;
                xend[rank][i] = FTProblem.nx;
                ystart[rank][i] = 1;
                yend[rank][i] = FTProblem.ny;
                zstart[rank][i] = 1;
                zend[rank][i] = FTProblem.nz;
            }
        } else if (FTProblem.layout_type == 1) {
            xstart[rank][0] = 1;
            xend[rank][0] = FTProblem.nx;
            ystart[rank][0] = 1;
            yend[rank][0] = FTProblem.ny;
            zstart[rank][0] = 1 + ((me2 * FTProblem.nz) / FTProblem.np2);
            zend[rank][0] = ((me2 + 1) * FTProblem.nz) / FTProblem.np2;

            xstart[rank][1] = 1;
            xend[rank][1] = FTProblem.nx;
            ystart[rank][1] = 1;
            yend[rank][1] = FTProblem.ny;
            zstart[rank][1] = 1 + ((me2 * FTProblem.nz) / FTProblem.np2);
            zend[rank][1] = ((me2 + 1) * FTProblem.nz) / FTProblem.np2;

            xstart[rank][2] = 1;
            xend[rank][2] = FTProblem.nx;
            ystart[rank][2] = 1 + ((me2 * FTProblem.ny) / FTProblem.np2);
            yend[rank][2] = ((me2 + 1) * FTProblem.ny) / FTProblem.np2;
            zstart[rank][2] = 1;
            zend[rank][2] = FTProblem.nz;
        } else if (FTProblem.layout_type == 2) {
            xstart[rank][0] = 1;
            xend[rank][0] = FTProblem.nx;
            ystart[rank][0] = 1 + ((me1 * FTProblem.ny) / FTProblem.np1);
            yend[rank][0] = ((me1 + 1) * FTProblem.ny) / FTProblem.np1;
            zstart[rank][0] = 1 + ((me2 * FTProblem.nz) / FTProblem.np2);
            zend[rank][0] = ((me2 + 1) * FTProblem.nz) / FTProblem.np2;

            xstart[rank][1] = 1 + ((me1 * FTProblem.nx) / FTProblem.np1);
            xend[rank][1] = ((me1 + 1) * FTProblem.nx) / FTProblem.np1;
            ystart[rank][1] = 1;
            yend[rank][1] = FTProblem.ny;
            zstart[rank][1] = zstart[rank][0];
            zend[rank][1] = zend[rank][0];

            xstart[rank][2] = xstart[rank][1];
            xend[rank][2] = xend[rank][1];
            ystart[rank][2] = 1 + ((me2 * FTProblem.ny) / FTProblem.np2);
            yend[rank][2] = ((me2 + 1) * FTProblem.ny) / FTProblem.np2;
            zstart[rank][2] = 1;
            zend[rank][2] = FTProblem.nz;
        }


    } // setup()
   
    private void init(int numberProcs){
    	chunks = new double[numberProcs][];
    	printStarted(FTProblem.kernelName, FTProblem.problemClassName, new long[] { FTProblem.nx, FTProblem.ny,
                    FTProblem.nz }, FTProblem.niter, numberProcs);
        if (FTProblem.layout_type==2){
            System.out.println("number of Processes > nz :"+numberProcs+">"+FTProblem.nz);
            System.out.println("Application won't work. Layout 2 not yet implemented");
            System.exit(1);
        }
        u0 = new double[numberProcs][];
        u1 = new double[numberProcs][];
        u2 = new double[numberProcs][];
        
        // data for kernel computation
        twiddle= new double[numberProcs][];
        u = new double[numberProcs][];
        xstart = new int[numberProcs][];
        xend = new int[numberProcs][];
        ystart = new int[numberProcs][];
        yend = new int[numberProcs][];
        zstart = new int[numberProcs][];
        zend = new int[numberProcs][];
        checkSumCount=1;
        z = new Complex[numberProcs][][];

        dims1 = new int[sizeDims * sizeDims];
       
        // allow to emulate fortran to pass by reference any section of an array
        // and take heed of the different memory allocation
        for (int j = 0; j < 3; j++) {
            dims1[0 * sizeDims + j] = FTProblem.dims[j+1][1]; // column 1
            dims1[1 * sizeDims + j] = FTProblem.dims[j+1][2]; // column 2
            dims1[2 * sizeDims + j] = FTProblem.dims[j+1][3]; // column 3
        }

        fftblock = NASFTImpl.fftblock_default;
        fftblockpad = NASFTImpl.fftblockpad_default;

        if (FTProblem.layout_type == 2) {
            if (FTProblem.dims[2][1] < fftblock) {
                fftblock = FTProblem.dims[2][1];
            }
            if (FTProblem.dims[2][2] < fftblock) {
                fftblock = FTProblem.dims[2][2];
            }
            if (FTProblem.dims[2][3] < fftblock) {
                fftblock = FTProblem.dims[2][3];
            }
        }

        if (fftblock != NASFTImpl.fftblock_default) {
            fftblockpad = fftblock + 3;
        }


        cplxSums = new double[FTProblem.niter + 1][2];
        cplxTemps = new double[numberProcs/4][numberProcs/4][16*2 * dataSize];
    }

    
    private void FFT(int numberProcs, char problemSize)
    throws Exception{
       if (layout_type == 0) {
            for (int rank=0;rank<numberProcs;rank++){
                NASFTImpl.FFT_layout0(numberProcs, problemSize, maxdim, dims1, u0[rank], u1[rank],  fftblock, fftblockpad, u[rank], twiddle[rank], xstart[rank][2], ystart[rank][1], ystart[rank][2], zstart[rank][1], zstart[rank][2]);
            }
        } else if (layout_type == 1) {


            double[][] scratch= new double[numberProcs][];
            for (int rank=0;rank<numberProcs;rank++){
                scratch[rank]= NASFTImpl.FFT_layout1_init(numberProcs,  problemSize, maxdim, dims[1][2] * dims[2][2], dims[3][2],
                										  dims1,
                										  u0[rank],
                										  u1[rank],
                										  fftblock, fftblockpad,
                										  u[rank],
                										  twiddle[rank],
                										  xstart[rank][2], ystart[rank][1], ystart[rank][2], zstart[rank][1], zstart[rank][2]);
            }

                if (numberProcs<16){
                    for (int receiver=0;receiver<numberProcs;receiver+=4){
                        for (int sender=0;sender<numberProcs;sender+=4){
                            NASFTImpl.getComplexArray(receiver, u0[sender], u0[sender+1], u0[sender+2], u0[sender+3], cplxTemps[receiver/4][sender/4]);
                        }
                    }

                    for (int sender=0;sender<numberProcs;sender+=4){
                        for (int receiver=0;receiver<numberProcs;receiver+=4){
                            NASFTImpl.setComplexArray(sender, cplxTemps[receiver/4][sender/4], u1[receiver], u1[receiver+1], u1[receiver+2], u1[receiver+3]);
                        }
                    }
               }else{
                    for (int receiver=0;receiver<numberProcs;receiver+=16){
                        for (int sender=0;sender<numberProcs;sender+=4){
                            for (int subrv=0;subrv<16;subrv+=4){
                                NASFTImpl.getComplexArray(receiver+subrv, u0[sender], u0[sender+1], u0[sender+2], u0[sender+3], cplxTemps[(receiver+subrv)/4][sender/4]);
                            }
                        }
                    }

                    for (int sender=0;sender<numberProcs;sender+=16){
                        for (int receiver=0;receiver<numberProcs;receiver+=4){
                            for (int subsender=0;subsender<16;subsender+=4){
                                NASFTImpl.setComplexArray((sender+subsender), cplxTemps[receiver/4][(sender+subsender)/4], u1[receiver], u1[receiver+1], u1[receiver+2], u1[receiver+3]);
                            }
                        }
                    }
               }
           
            for (int rank=0;rank<numberProcs;rank++){
                NASFTImpl.FFT_layout1_end(np2, dims[1][2]* dims[2][2], dims[3][2],
                						  dims1,
                						  u0[rank],
                						  u1[rank],
                						  scratch[rank],
                						  fftblock, fftblockpad,
                						  u[rank]);
           }
            
        }
    }

    
    public void inverseFFT(int numberProcs, int iteration)
    throws Exception{
           if (layout_type == 0) {
                for (int rank=0;rank<numberProcs;rank++){
                    chunks[rank]=NASFTImpl.inverseFFT_layout0(nx, ny, nz, (int)ntotal_f, maxdim, dims1, u0[rank], u1[rank], u2[rank], twiddle[rank], fftblock, fftblockpad, u[rank], xstart[rank][0], xend[rank][0], ystart[rank][0], yend[rank][0], zstart[rank][0], zend[rank][0]);
                }
            } else if (layout_type == 1) {
            	double[][] scratch= new double[numberProcs][];
                for (int rank=0;rank<numberProcs;rank++){
                    scratch[rank]= NASFTImpl.inverseFFT_layout1_init(maxdim, dims[1][3], dims[2][3] * dims[3][3], dims1,
                    												 u0[rank],
                    												 u1[rank],
                    												 u2[rank],
                    												 twiddle[rank],
                    												 fftblock, fftblockpad,
                    												 u[rank]);
                }
                if (numberProcs<16){
                    for (int receiver=0;receiver<numberProcs;receiver+=4){
                        for (int sender=0;sender<numberProcs;sender+=4){
                            NASFTImpl.getComplexArray(receiver, u2[sender], u2[sender+1], u2[sender+2], u2[sender+3], cplxTemps[receiver/4][sender/4]);
                        }
                    }

                    for (int sender=0;sender<numberProcs;sender+=4){
                        for (int receiver=0;receiver<numberProcs;receiver+=4){
                            NASFTImpl.setComplexArray(sender, cplxTemps[receiver/4][sender/4], u1[receiver], u1[receiver+1], u1[receiver+2], u1[receiver+3]);
                        }
                    }
               }else{
                    for (int receiver=0;receiver<numberProcs;receiver+=16){
                        for (int sender=0;sender<numberProcs;sender+=4){
                            for (int subrv=0;subrv<16;subrv+=4){
                                NASFTImpl.getComplexArray(receiver+subrv, u2[sender], u2[sender+1], u2[sender+2], u2[sender+3], cplxTemps[(receiver+subrv)/4][sender/4]);
                            }
                        }
                    }

                    for (int sender=0;sender<numberProcs;sender+=16){
                        for (int receiver=0;receiver<numberProcs;receiver+=4){
                            for (int subsender=0;subsender<16;subsender+=4){
                                NASFTImpl.setComplexArray((sender+subsender), cplxTemps[receiver/4][(sender+subsender)/4], u1[receiver], u1[receiver+1], u1[receiver+2], u1[receiver+3]);
                            }
                        }
                    }
               }
                for (int rank=0;rank<numberProcs;rank++){
                    chunks[rank]=NASFTImpl.inverseFFT_layout1_end(nx, ny, nz, np2, ntotal_f, dims[1][3], dims[2][3] * dims[3][3],
                    											  dims1,
                    											  u1[rank],
                    											  u2[rank],
                    											  scratch[rank],
                    											  fftblock, fftblockpad,
                    											  u[rank],
                    											  xstart[rank][0], xend[rank][0], ystart[rank][0], yend[rank][0], zstart[rank][0], zend[rank][0]);
               }
           }
    }
    
    
    private void reduction(int numberProcs) {
    	int neighbor=1;
        while (neighbor<numberProcs){
            for (int i=0;i<numberProcs;i+=2*neighbor){
                if (i+neighbor<numberProcs){
                	NASFTImpl.reduceComplex(chunks[i], chunks[i+neighbor]);    
                }
            }
            neighbor*=2;
        }
    	
    	checkSumCount++;
    	NASFTImpl.reduceComplex(cplxSums[checkSumCount-1], chunks[0]);
    }
    
    private void allocation(int numberProcs) {
    	System.out.println("SIZES in MBytes: ");
    	System.out.println("- Complex Arrays: " + (((dataSize * 2) * 8) / 1048576));
    	System.out.println("- Complex Array Groups: " + ((8*(4*FTProblem.ntdivnp))/1048576));
    	System.out.println("- Twiddle: " + ((8*(FTProblem.ntdivnp))/1048576));
    	System.out.println("- U: " + ((8*(2*(FTProblem.nx + 1)))/1048576));
    	System.out.println("- Scratch: " + ((8*(18 * maxdim * 4))/1048576));
    	
    	
    	for (int rank=0;rank<numberProcs;rank++){
            u0[rank] = NASFTImpl.allocComplexArray(4*FTProblem.ntdivnp);
        }
    	for (int rank=0;rank<numberProcs;rank++){
            u1[rank] = NASFTImpl.allocComplexArray(4*FTProblem.ntdivnp);
        }
    	for (int rank=0;rank<numberProcs;rank++){
            u2[rank] = NASFTImpl.allocComplexArray(4*FTProblem.ntdivnp);
        }
    	for (int rank=0;rank<numberProcs;rank++){
            twiddle[rank] = NASFTImpl.allocComplexArray(FTProblem.ntdivnp);
        }
    	for (int rank=0;rank<numberProcs;rank++){
            u[rank] = NASFTImpl.allocComplexArray(2*(FTProblem.nx + 1));
        }
    }
}

