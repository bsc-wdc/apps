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

public class NASISImpl {

    /**
     * Generate locally a sequence of key with the NAS Random generator Parameters must be odd
     * double precision integers in the range (1, 2^46)
     *
     * @param seed
     * @param a
     */

    public static int[] create_seq(
            double seed,
            double a,
            int rank,
            ISProblemClass ISProblem) {
        int[] key_array = new int[ISProblem.buffersSize];
        Random rng = new Random();
        double x;
        int k;
        int forTest;
        k = ISProblem.maxKey / 4;

        rng.setSeed(seed);
        rng.setGmult(a);

        forTest = ISProblem.numKeys * rank;
        for (int i = 0; i < forTest; i++) {
            rng.randlc();
            rng.randlc();
            rng.randlc();
            rng.randlc();
        }

        for (int i = 0; i < ISProblem.numKeys; i++) {
            x = rng.randlc();
            x += rng.randlc();
            x += rng.randlc();
            x += rng.randlc();
            key_array[i] = (int) (x * k);
        }
        return key_array;
    }

    public static int[] rank0(
            ISProblemClass ISProblem,
            int iteration,
            int[] key_array,
            int[] key_buff1) {
        int i, j;
        int[] t;
        int shift = ISProblem.maxKeyLog2 - ISProblem.numBucketsLog2;
        int key;
        int arraySize = ISProblem.numBuckets + ISProblem.testArraySize;
        int[] bucket_size = new int[arraySize];
        /* Iteration alteration of keys */

        key_array[iteration] = iteration;
        key_array[iteration + ISProblem.maxIterations] = ISProblem.maxKey - iteration;


        /* Initialize */
        // ~1000 loops
        j = ISProblem.numBuckets + ISProblem.testArraySize;
        for (i = 0; i < j; i++) {
            bucket_size[i] = 0;
        }

        /*
         * Determine where the partial verify test keys are, load into top of array bucket_size
         */
        // ~5 loops
        t = ISProblem.testIndexArray;
        for (i = 0; i < ISProblem.testArraySize; i++) {
            if ((t[i] / ISProblem.numKeys) == 0) {
                bucket_size[ISProblem.numBuckets + i] = key_array[t[i] % ISProblem.numKeys];
            }
        }

        /* Determine the number of keys in each bucket */
        // Up to 2^29 loops
        j = ISProblem.numKeys;
        for (i = 0; i < j; i++) {
            bucket_size[key_array[i] >> shift]++;
        }

        /* Accumulative bucket sizes are the bucket pointers */
        // ~1000 loops
        int[] bucket_ptrs = new int[ISProblem.numBuckets];
        bucket_ptrs[0] = 0;
        for (i = 1; i < ISProblem.numBuckets; i++) {
            bucket_ptrs[i] = bucket_ptrs[i - 1] + bucket_size[i - 1];
        }

        /* Sort into appropriate bucket */
        // Up to 2^29 loops
        j = ISProblem.numKeys;
        for (i = 0; i < j; i++) {
            key = key_array[i];
            key_buff1[bucket_ptrs[key >> shift]++] = key;
        }
        return bucket_size;
    }

    public static int[] rank(
            ISProblemClass ISProblem,
            int iteration,
            int rank,
            int[] key_array,
            int[] key_buff1) {
        int arraySize = ISProblem.numBuckets + ISProblem.testArraySize;
        int[] bucket_size = new int[arraySize];
        int i, j;
        int[] t;
        int shift = ISProblem.maxKeyLog2 - ISProblem.numBucketsLog2;
        int key;

        /* Initialize */
        // ~1000 loops
        j = ISProblem.numBuckets + ISProblem.testArraySize;
        for (i = 0; i < j; i++) {
            bucket_size[i] = 0;
        }

        /*
         * Determine where the partial verify test keys are, load into top of array bucket_size
         */
        // ~5 loops
        t = ISProblem.testIndexArray;
        for (i = 0; i < ISProblem.testArraySize; i++) {
            if ((t[i] / ISProblem.numKeys) == rank) {
                bucket_size[ISProblem.numBuckets + i] = key_array[t[i] % ISProblem.numKeys];
            }
        }

        /* Determine the number of keys in each bucket */
        // Up to 2^29 loops
        j = ISProblem.numKeys;
        for (i = 0; i < j; i++) {
            bucket_size[key_array[i] >> shift]++;
        }

        /* Accumulative bucket sizes are the bucket pointers */
        // ~1000 loops
        int[] bucket_ptrs = new int[ISProblem.numBuckets];
        bucket_ptrs[0] = 0;
        for (i = 1; i < ISProblem.numBuckets; i++) {
            bucket_ptrs[i] = bucket_ptrs[i - 1] + bucket_size[i - 1];
        }

        /* Sort into appropriate bucket */
        // Up to 2^29 loops
        j = ISProblem.numKeys;
        for (i = 0; i < j; i++) {
            key = key_array[i];
            key_buff1[bucket_ptrs[key >> shift]++] = key;
        }
        return bucket_size;
    }

    public static int[] reduceBucketSize(ISProblemClass ISProblem, int[] bucket_size1, int[] bucket_size2){
        int arraySize = ISProblem.numBuckets + ISProblem.testArraySize;
        int[] bucket_size_totals = new int[arraySize];
        for (int i = 0; i < arraySize; i++) {
            bucket_size_totals[i] = bucket_size1[i]+bucket_size2[i];
        }
        return bucket_size_totals;
    }

    public static void prepareSend(ISProblemClass ISProblem, int[] process_bucket_distrib_ptr1, int[] process_bucket_distrib_ptr2, int[] bucket_size_totals, int[] bucket_size, int[] send_count) {
        int i, j;
        int bucket_sum_accumulator;
        int local_bucket_sum_accumulator;

        bucket_sum_accumulator = 0;
        local_bucket_sum_accumulator = 0;
        process_bucket_distrib_ptr1[0] = 0;
        int numBuckets = ISProblem.numBuckets;
        int numKeys = ISProblem.numKeys;

        for (i = 0, j = 0; i < numBuckets; i++) {
            bucket_sum_accumulator += bucket_size_totals[i];
            local_bucket_sum_accumulator += bucket_size[i];

            if (bucket_sum_accumulator >= ((j + 1) * numKeys)) {
                send_count[j] = local_bucket_sum_accumulator;

                if (j != 0) {
                    process_bucket_distrib_ptr1[j] = process_bucket_distrib_ptr2[j - 1] + 1;
                }

                process_bucket_distrib_ptr2[j++] = i;
                local_bucket_sum_accumulator = 0;
            }
        }
    }


    public static void rank_end(ISProblemClass ISProblem, int rank, int[] process_bucket_distrib_ptr1, int[] process_bucket_distrib_ptr2, int[] key_buff1, int[] key_buff2, int[] bucket_size_totals, int iteration, int[] verifies, int[] total_local_keys) {
    	int i;
        int j;
        int k;
        int m;
        int temp;
        int shift = ISProblem.maxKeyLog2 - ISProblem.numBucketsLog2;
        int min_key_val;
        int max_key_val;

        min_key_val = process_bucket_distrib_ptr1[rank] << shift;
        max_key_val = ((process_bucket_distrib_ptr2[rank] + 1) << shift) - 1;
        
        for (i = 0; i < max_key_val - min_key_val + 1; i++) {
            key_buff1[i] = 0;
        }
        m = 0;
        for (k = 0; k < rank; k++) {
            for (i = process_bucket_distrib_ptr1[k]; i <= process_bucket_distrib_ptr2[k]; i++) {
                m += bucket_size_totals[i];
            }
        }
        j = 0;
        for (i = process_bucket_distrib_ptr1[rank]; i <= process_bucket_distrib_ptr2[rank]; i++) {
            j += bucket_size_totals[i];
        }
        for (i = 0; i < j; i++) {
            key_buff1[key_buff2[i] - min_key_val]++;
        }
        key_buff1[0] += m;
        temp = (max_key_val - min_key_val);
        for (i = 0; i < temp; i++) {
            key_buff1[i + 1] += key_buff1[i];
        }

        verifies[0]=partialVerify(ISProblem, min_key_val, max_key_val, rank, iteration, key_buff1, bucket_size_totals);
        
        total_local_keys[0] = j;
        total_local_keys[1] = m;
        total_local_keys[2] = min_key_val;
    }

    
    public static void sortKeys(int[] total_local_keys, int[] key_buff1, int[] key_buff2, int[] key_array) {
    	int j = total_local_keys[0];
    	int m = total_local_keys[1];
    	int min_key_val = total_local_keys[2];
    	
    	/* Now, finally, sort the keys: */
    	int key;
        for (int i = 0; i < j; i++) {
        	key = key_buff2[i];
            key_array[--key_buff1[key - min_key_val] - m] = key;
        }
    }


    public static int[] get_keypart(int[] key_buff10, int[] key_buff11, int[] key_buff12, int[] key_buff13, int[] send_count0, int[] send_count1, int[] send_count2, int[] send_count3, int receiver) {
        int sent_count0=0;
        int sent_count1=0;
        int sent_count2=0;
        int sent_count3=0;

        for (int i=0;i<receiver;i++){
            sent_count0+=send_count0[i];
            sent_count1+=send_count1[i];
            sent_count2+=send_count2[i];
            sent_count3+=send_count3[i];
        }
        
        
        int toSendCount0=send_count0[receiver]+send_count1[receiver]+send_count2[receiver]+send_count3[receiver];
        int toSendCount1=send_count0[receiver+1]+send_count1[receiver+1]+send_count2[receiver+1]+send_count3[receiver+1];
        int toSendCount2=send_count0[receiver+2]+send_count1[receiver+2]+send_count2[receiver+2]+send_count3[receiver+2];
        int toSendCount3=send_count0[receiver+3]+send_count1[receiver+3]+send_count2[receiver+3]+send_count3[receiver+3];


        int[] key_buff= new int[4+toSendCount0+toSendCount1+toSendCount2+toSendCount3];

        key_buff[0]=toSendCount0;
        key_buff[1]=toSendCount1;
        key_buff[2]=toSendCount2;
        key_buff[3]=toSendCount3;

        int totalsend=4;
        for (int i=0;i<4;i++){

            System.arraycopy(key_buff10, sent_count0, key_buff, totalsend, send_count0[receiver+i]);
            totalsend+=send_count0[receiver+i];
            sent_count0+=send_count0[receiver+i];
            System.arraycopy(key_buff11, sent_count1, key_buff, totalsend, send_count1[receiver+i]);
            totalsend+=send_count1[receiver+i];
            sent_count1+=send_count1[receiver+i];
            System.arraycopy(key_buff12, sent_count2, key_buff, totalsend, send_count2[receiver+i]);
            totalsend+=send_count2[receiver+i];
            sent_count2+=send_count2[receiver+i];
            System.arraycopy(key_buff13, sent_count3, key_buff, totalsend, send_count3[receiver+i]);
            totalsend+=send_count3[receiver+i];
            sent_count3+=send_count3[receiver+i];
        }
        
        return key_buff;
    }


    public static void transferKeys(int[] key_buff, int[] key_buff20, int[] key_buff21, int[] key_buff22, int[] key_buff23, int[] recv_displ0,int[] recv_displ1,int[] recv_displ2,int[] recv_displ3) {
        System.arraycopy(key_buff, 4, key_buff20, recv_displ0[0], key_buff[0]);
        System.arraycopy(key_buff, 4+key_buff[0], key_buff21, recv_displ1[0], key_buff[1]);
        System.arraycopy(key_buff, 4+key_buff[0]+key_buff[1], key_buff22, recv_displ2[0], key_buff[2]);
        System.arraycopy(key_buff, 4+key_buff[0]+key_buff[1]+key_buff[2], key_buff23, recv_displ3[0], key_buff[3]);
        recv_displ0[0]+=key_buff[0];
        recv_displ1[0]+=key_buff[1];
        recv_displ2[0]+=key_buff[2];
        recv_displ3[0]+=key_buff[3];
    }


    private static int partialVerify(ISProblemClass ISProblem, int min_key_val, int max_key_val, int rank, int iteration, int[] key_buff1, int[] bucket_size_totals) {
    int passed_verification=0;
        /* This is the partial verify test section */
        /* Observe that test_rank_array vals are */
        /* shifted differently for different cases */
        for (int i = 0; i < ISProblem.testArraySize; i++) {
            int k = bucket_size_totals[i + ISProblem.numBuckets]; /* Keys were hidden here */
            if ((min_key_val <= k) && (k <= max_key_val)) {
                switch (ISProblem.problemClassName) {
                    case 'S':
                        if (i <= 2) {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] + iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        } else {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] - iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        }
                        break;
                    case 'W':
                        if (i < 2) {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] + (iteration - 2))) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        } else {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] - iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        }
                        break;
                    case 'A':
                        if (i <= 2) {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] + (iteration - 1))) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        } else {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] - (iteration - 1))) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        }
                        break;
                    case 'B':
                        if ((i == 1) || (i == 2) || (i == 4)) {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] + iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        } else {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] - iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        }
                        break;
                    case 'C':
                        if (i <= 2) {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] + iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        } else {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] - iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                            } else {
                                passed_verification++;
                            }
                        }
                        break;
                    case 'D':
                        if (i <= 2) {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] + iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                                System.out.println("test_rank_array[" + i + "]: " +
                                    (key_buff1[k - 1 - min_key_val] - iteration));
                            } else {
                                passed_verification++;
                            }

                            System.out.println("Verified: proc " + rank + ", test_rank_array[" + i +
                                "]: " + (key_buff1[k - 1 - min_key_val] - iteration));
                        } else {
                            if (key_buff1[k - 1 - min_key_val] != (ISProblem.testRankArray[i] - iteration)) {
                                System.out.println("Failed partial verification: " + "iteration " +
                                    iteration + ", processor " + rank + ", test key " + i);
                                System.out.println("test_rank_array[" + i + "]: " +
                                    (key_buff1[k - 1 - min_key_val] + iteration));
                            } else {
                                passed_verification++;
                            }

                            System.out.println("Verified: proc " + rank + ", test_rank_array[" + i +
                                "]: " + (key_buff1[k - 1 - min_key_val] + iteration));
                        }
                        break;
                }
            }else{
                passed_verification++;
            }
        }
    return passed_verification;
    }

    public static int synchronize(int[] value) {
        return 1;
    }

    public static int[] allocIntArray(int buffersSize) {
        return new int[buffersSize];
    }

}
