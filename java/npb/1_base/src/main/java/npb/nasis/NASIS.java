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
package npb.nasis;


public class NASIS {

    static ISProblemClass ISProblem;
    static int[][] key_array, key_buff1, key_buff2;
    static int[][] bucket_size, process_bucket_distrib_ptr1, process_bucket_distrib_ptr2;
    static int[][] send_count;
    static int[] bucket_size_totals;
    int[] passed_verification;
    int[][] total_local_keys;

    long initTime;

    public static void main(String[] args){
        char problemSize = 'B';
        int np = 16;
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
        new NASIS(problemSize, np);

    }
    
    boolean verified;
    int arraySize;
    int maxIterations;
    int[][][] verifies;
    int[][] reductions;
    int[][][] keycache;
    NASIS(char problemSize, int numberProcs) {
        try{
        verified=true;
        ISProblem = ClassFactory.getNASISClass(numberProcs, problemSize);
        Verifier.ISProblem=ISProblem;
        Reducer.ISProblem=ISProblem;
        maxIterations=ISProblem.maxIterations;
        verifies = new int[numberProcs][maxIterations+1][1];
        keycache=new int[numberProcs/4][numberProcs/4][];
        
        key_array = new int[ISProblem.numProcs][];
        key_buff1 = new int[ISProblem.numProcs][];
        key_buff2 = new int[ISProblem.numProcs][];

        reductions = new int[numberProcs][];

        arraySize = ISProblem.numBuckets + ISProblem.testArraySize;
        
        bucket_size = new int[ISProblem.numProcs][];

        process_bucket_distrib_ptr1 = new int[ISProblem.numProcs][];
        process_bucket_distrib_ptr2 = new int[ISProblem.numProcs][];

        send_count = new int[ISProblem.numProcs][ISProblem.numProcs];

        
        total_local_keys =  new int[ISProblem.numProcs][3];
        passed_verification= new int[ISProblem.numProcs];

        for (int rank = 0; rank < numberProcs; rank++) {
            key_array[rank] = NASISImpl.create_seq(314159265.00d, 1220703125.00d, rank, ISProblem);
        }
        for (int rank = 0; rank < numberProcs; rank++) {
            key_buff1[rank] = NASISImpl.allocIntArray(ISProblem.buffersSize);
        }
        for (int rank = 0; rank < numberProcs; rank++) {
            key_buff2[rank] = NASISImpl.allocIntArray(ISProblem.buffersSize);
        }
        printStarted(ISProblem.problemClassName, ISProblem.totalKeys, ISProblem.maxIterations, numberProcs);

        
        // WARM UP
        rank(numberProcs, 1, arraySize);
        exchange(numberProcs);
        rankEnd(numberProcs, 1);
        
        for (int rank=0;rank<numberProcs;rank++){
            NASISImpl.synchronize(verifies[rank][1]);
        }

        initTime=System.currentTimeMillis();

        for (int iteration=1;iteration <= maxIterations;iteration++){
            rank(numberProcs, iteration, arraySize);
            exchange(numberProcs);
            rankEnd(numberProcs, iteration);
            System.out.println("        " + iteration);
        }
        
        for (int rank=0;rank<numberProcs;rank++){
            NASISImpl.synchronize(verifies[rank][maxIterations]);
        }

        initTime=System.currentTimeMillis()-initTime;
        System.out.println("End Time:"+initTime);
        
        for (int rank=0;rank<numberProcs;rank++) {
        	NASISImpl.sortKeys(total_local_keys[rank], key_buff1[rank], key_buff2[rank], key_array[rank]);
        }


        passed_verification[0]+=Verifier.full_verify2(-1,0,total_local_keys[0], key_array[0]);
        for (int rank=1;rank<ISProblem.numProcs;rank++){
            int k=Verifier.getK(key_array[rank-1], total_local_keys[rank-1]);
            passed_verification[rank]+=Verifier.full_verify2(k, rank, total_local_keys[rank], key_array[rank]);
        }
        
        for (int rank=0;rank<ISProblem.numProcs;rank++){
            for (int it=1;it<maxIterations;it++){
                verified=verified&&Verifier.parcial_verify(verifies[rank][it]);
            }

            verified=verified&&Verifier.final_verify(passed_verification[rank]);
        }
        printEnd((double)initTime/(double)1000, getMflops(initTime), verified);

        }catch(Exception e){
            e.printStackTrace();
        }
    }





















   
    public void printStarted(char className, long size, int nbIteration, int nbProcess) {
        System.out.print("\n\n NAS Parallel Benchmarks ProActive -- IS Benchmark\n\n");
        System.out.println(" Class: " + className);
        System.out.print(" Size:  " + size);
        System.out.println(" Iterations:   " + nbIteration);
        System.out.println(" Number of processes:     " + nbProcess);
    }

    public void printEnd(double totalTime, double mops, boolean passed_verification) {
        String verif;
        String javaVersion = System.getProperty("java.vm.vendor") + " " + System.getProperty("java.vm.name")
                + " " + System.getProperty("java.vm.version") + " - Version " + System.getProperty("java.version");

        verif = passed_verification ? "SUCCESSFUL" : "UNSUCCESSFUL";

        System.out.println("\n\n " + ISProblem.kernelName + " Benchmark Completed");
        System.out.println(" Class            =  " + ISProblem.problemClassName);
        System.out.println(" Size             =  " + ISProblem.size);
        System.out.println(" Iterations       =  " + ISProblem.iterations);
        System.out.println(" Time in seconds  =  " + totalTime);
        System.out.println(" Total processes  =  " + ISProblem.numProcs);
        System.out.println(" Mop/s total      =  " + mops);
        System.out.println(" Mop/s/process    =  " + (mops / ISProblem.numProcs));
        System.out.println(" Operation type   =  " + ISProblem.operationType);
        System.out.println(" Verification     =  " + verif);
        System.out.println(" NPB Version      =  " + ISProblem.version);
        System.out.println(" Java RE          =  " + javaVersion);
    }


    private double getMflops(long total) {
        double time = total / 1000.0;
        double mflops = (double) (ISProblem.maxIterations * ISProblem.totalKeys) / time / 1000000.0;
        return mflops;
    }




    private void rank(int numberProcs, int iteration, int arraySize){
        bucket_size[0]=NASISImpl.rank0(ISProblem, iteration, key_array[0], key_buff1[0]);
            process_bucket_distrib_ptr1[0] = new int[arraySize];
            process_bucket_distrib_ptr2[0] = new int[arraySize];

        for (int rank = 1; rank < numberProcs; rank++) {
            bucket_size[rank]=NASISImpl.rank(ISProblem, iteration, rank, key_array[rank], key_buff1[rank]);
            process_bucket_distrib_ptr1[rank] = new int[arraySize];
            process_bucket_distrib_ptr2[rank] = new int[arraySize];
        }
    }
    
    private void exchange(int numberProcs){
        /* REDUCTION ON WORKERS*/

        for (int i=0;i<numberProcs;i++){
            reductions[i]=bucket_size[i];
        }
        int neighbor=1;
        while (neighbor<numberProcs){
            for (int i=0;i<numberProcs;i+=2*neighbor){
                if (i+neighbor<numberProcs){
                    reductions[i]=NASISImpl.reduceBucketSize(ISProblem, reductions[i], reductions[i+neighbor]);
                }
            }
            neighbor*=2;
        }
        bucket_size_totals= reductions[0];


        for (int rank = 0; rank < numberProcs; rank++) {
            NASISImpl.prepareSend(ISProblem, process_bucket_distrib_ptr1[rank], process_bucket_distrib_ptr2[rank], bucket_size_totals, bucket_size[rank], send_count[rank]);
        }

        if (numberProcs<16){
            for (int rank = 0; rank < numberProcs; rank+=4) {
                for (int sender = 0; sender < numberProcs; sender+=4) {
                    keycache[sender/4][rank/4]=NASISImpl.get_keypart(key_buff1[sender], key_buff1[sender+1], key_buff1[sender+2], key_buff1[sender+3], send_count[sender], send_count[sender+1], send_count[sender+2], send_count[sender+3],rank);
                }
            }
        }else{
            for (int rank = 0; rank < numberProcs; rank+=16) {
                for (int sender = 0; sender < numberProcs; sender+=4) {
                    for (int subrank = 0; subrank < 16; subrank+=4) {
                        keycache[sender/4][(rank+subrank)/4]=NASISImpl.get_keypart(key_buff1[sender], key_buff1[sender+1], key_buff1[sender+2], key_buff1[sender+3], send_count[sender], send_count[sender+1], send_count[sender+2], send_count[sender+3],rank+subrank);
                    }
                }
            }
        }

        int[][] recvCount= new int[numberProcs][1];

        if (numberProcs<16){
            for (int sender = 0; sender < numberProcs; sender+=4) {
                for (int rank = 0; rank < numberProcs; rank+=4) {
                    NASISImpl.transferKeys(keycache[sender/4][rank/4], key_buff2[rank], key_buff2[rank+1], key_buff2[rank+2], key_buff2[rank+3], recvCount[rank], recvCount[rank+1], recvCount[rank+2], recvCount[rank+3]);
                }
            }
        }else{
            for (int sender = 0; sender < numberProcs; sender+=16) {
                for (int rank = 0; rank < numberProcs; rank+=4) {
                    for (int subsender = 0; subsender < 16; subsender+=4) {
                        NASISImpl.transferKeys(keycache[(sender+subsender)/4][rank/4], key_buff2[rank], key_buff2[rank+1], key_buff2[rank+2], key_buff2[rank+3], recvCount[rank], recvCount[rank+1], recvCount[rank+2], recvCount[rank+3]);
                    }
                }
            }
        }
    }

    private void rankEnd(int numberProcs, int iteration){
        //rank_end
        for (int rank = 0; rank < numberProcs; rank++) {
            NASISImpl.rank_end(ISProblem, rank, process_bucket_distrib_ptr1[rank], process_bucket_distrib_ptr2[rank], key_buff1[rank], key_buff2[rank], bucket_size_totals, iteration, verifies[rank][iteration], total_local_keys[rank]);
        }
    }

}


