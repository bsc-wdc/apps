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

package npb.nasep;

/**
 *
 * @author flordan
 */
public class NASEP {

    private static EPProblemClass EPProblem;
    
    
    private double[] values;

    public static void main(String[] args) {
        char problemSize='D';
        int np=16;
        for (int i = 0; i < args.length; i++) {
            System.out.println(args[i]);
            if (args[i].compareToIgnoreCase("-np") == 0) {
            np=Integer.parseInt(args[++i]);
            } else if (args[i].compareToIgnoreCase("-class") == 0) {
                if (args[++i].length() == 1) {
                    problemSize=args[i].toUpperCase().charAt(0);
                } else {
                    System.err
                            .println("Parse args error: " + args[i - 1] + " must have a one char parameter");
                    System.exit(1);
                }
            }
        }
        new NASEP(problemSize, np);
        
    }
    
    
    NASEP(char problemSize, int numberProcs){
        try{
        long time;


        values = new double[3];

        EPProblem=getEPNASClass(problemSize, numberProcs);

        printStarted(EPProblem.problemClassName, new long[] { EPProblem.m }, 0, EPProblem.numProcs);
        
        double[][] epvs= new double[numberProcs][];

        time=System.currentTimeMillis();
        int m=EPProblem.m;
        
        for (int rank=0;rank<numberProcs;rank++){
            epvs[rank]=NASEPImpl.generate(m, numberProcs, rank);
        }

                    /* REDUCTION ON WORKERS*/
            int neighbor=1;
            while (neighbor<numberProcs){
                for (int i=0;i<numberProcs;i+=2*neighbor){
                    if (i+neighbor<numberProcs){
                        NASEPImpl.reduce(epvs[i], epvs[i+neighbor]);
                    }
                }
                neighbor*=2;
            }
           values = Synchronizer.getValues(epvs[0]);

        /* REDUCTION ON MASTER */
        /*for (int rank=0;rank<numberProcs;rank++){
            Reducer.reduce(values, epvs[rank]);
        }*/
        
        time=System.currentTimeMillis()-time;
        printEnd(time, getMflops(time),  verify());

        }catch(Exception e){
            e.printStackTrace();
        }
    }

    

    
    public double getMflops(long totalTime) {
        double timeInSec = totalTime / 1000.0;
        double mflops = Math.pow(2.0, EPProblem.m + 1) / timeInSec / 1000000.0;
        return mflops;
    }
    
    private EPProblemClass getEPNASClass(char clss, int np) {
        EPProblemClass cl = new EPProblemClass();

        switch (clss) {
            case 'S':
                cl.m = 24;
                break;
            case 'W':
                cl.m = 25;
                break;
            case 'A':
                cl.m = 28;
                break;
            case 'B':
                cl.m = 30;
                break;
            case 'C':
                cl.m = 32;
                break;
            case 'D':
                cl.m = 36;
        }

        //common variables
        cl.kernelName = "EP";
        cl.problemClassName = clss;
        cl.operationType = "Random numbers generated";
        cl.numProcs = np;
        cl.npm = np;
        cl.iterations = 1;
        cl.size = cl.m;
        cl.sizeStr = "" + cl.m;
        cl.version = "3.2";

        
        
        return cl;
    }


    public void printStarted(char className, long[] size, int nbIteration, int nbProcess) {
        System.out.print("\n\n NAS Parallel Benchmarks ProActive -- EP Benchmark\n\n");
        System.out.println(" Class: " + className);
        System.out.print(" Size:  " + size[0]);
        for (int i = 1; i < size.length; i++) {
            System.out.print(" x " + size[i]);
        }
        System.out.println();

        System.out.println(" Iterations:   " + nbIteration);
        System.out.println(" Number of processes:     " + nbProcess);
    }

    public void printEnd(double totalTime, double mops,  boolean passed_verification) {
        String verif;
        String javaVersion = System.getProperty("java.vm.vendor") + " " + System.getProperty("java.vm.name") +
            " " + System.getProperty("java.vm.version") + " - Version " + System.getProperty("java.version");

        verif = passed_verification ? "SUCCESSFUL" : "UNSUCCESSFUL";

        System.out.println("\n\n " + EPProblem.kernelName + " Benchmark Completed");
        System.out.println(" Class            =  " + EPProblem.problemClassName);
        System.out.println(" Size             =  " + EPProblem.size);
        System.out.println(" Iterations       =  " + EPProblem.iterations);
        System.out.println(" Time in seconds  =  " + totalTime/1000);
        System.out.println(" Total processes  =  " + EPProblem.numProcs);
        System.out.println(" Mop/s total      =  " + mops);
        System.out.println(" Mop/s/process    =  " + (mops / EPProblem.numProcs));
        System.out.println(" Operation type   =  " + EPProblem.operationType);
        System.out.println(" Verification     =  " + verif);
        System.out.println(" NPB Version      =  " + EPProblem.version);
        System.out.println(" Java RE          =  " + javaVersion);
    }
    
    private boolean verify() {
        double epsilon = .00000001d;
        boolean verified = false;

        switch (EPProblem.m) {
            case 24:
                verified = ((Math.abs((values[0] - (-3247.834652034740)) / values[0]) <= epsilon) && (Math
                        .abs((values[1] - (-6958.407078382297)) / values[1]) <= epsilon));
                break;
            case 25:
                verified = ((Math.abs((values[0] - (-2863.319731645753)) / values[0]) <= epsilon) && (Math
                        .abs((values[1] - (-6320.053679109499)) / values[1]) <= epsilon));
                break;
            case 28:
                verified = ((Math.abs((values[0] - (-4295.875165629892)) / values[0]) <= epsilon) && (Math
                        .abs((values[1] - (-15807.32573678431)) / values[1]) <= epsilon));
                break;
            case 30:
                verified = ((Math.abs((values[0] - (40338.15542441498)) / values[0]) <= epsilon) && (Math
                        .abs((values[1] - (-26606.69192809235)) / values[1]) <= epsilon));
                break;
            case 32:
                verified = ((Math.abs((values[0] - (47643.67927995374)) / values[0]) <= epsilon) && (Math
                        .abs((values[1] - (-80840.72988043731)) / values[1]) <= epsilon));
                break;
            case 36:
                verified = ((Math.abs((values[0] - (198248.1200946593)) / values[0]) <= epsilon) && (Math
                        .abs((values[1] - (-102059.6636361769)) / values[1]) <= epsilon));
                break;
        }

        return verified;
    }

    
}
