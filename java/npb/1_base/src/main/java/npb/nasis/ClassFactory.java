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


public class ClassFactory {

    public static ISProblemClass getNASISClass(int np, char clss){
        ISProblemClass cl = new ISProblemClass();

        switch (clss) {
            case 'S':
                cl.totalKeysLog2 = 16;
                cl.maxKeyLog2 = 11;
                cl.numBucketsLog2 = 9;
                int[] S_test_index_array = {48427, 17148, 23627, 62548, 4431};
                int[] S_test_rank_array = {0, 18, 346, 64917, 65463};
                cl.testIndexArray = S_test_index_array;
                cl.testRankArray = S_test_rank_array;
                break;
                
            case 'W':
                cl.totalKeysLog2 = 20;
                cl.maxKeyLog2 = 16;
                cl.numBucketsLog2 = 10;
                cl.testIndexArray = new int[] {357773, 934767, 875723, 898999, 404505};
                cl.testRankArray = new int[] {1249, 11698, 1039987, 1043896, 1048018};
                break;
                
            case 'A':
                cl.totalKeysLog2 = 23;
                cl.maxKeyLog2 = 19;
                cl.numBucketsLog2 = 10;
                int[] A_test_index_array = {2112377, 662041, 5336171, 3642833, 4250760};
                int[] A_test_rank_array = {104, 17523, 123928, 8288932, 8388264};
                cl.testIndexArray = A_test_index_array;
                cl.testRankArray = A_test_rank_array;
                break;
                
            case 'B':
                cl.totalKeysLog2 = 25;
                cl.maxKeyLog2 = 21;
                cl.numBucketsLog2 = 10;
                int[] B_test_index_array = {41869, 812306, 5102857, 18232239, 26860214};
                int[] B_test_rank_array = {33422937, 10244, 59149, 33135281, 99};
                cl.testIndexArray = B_test_index_array;
                cl.testRankArray = B_test_rank_array;
                break;
                
            case 'C':
                cl.totalKeysLog2 = 27;
                cl.maxKeyLog2 = 23;
                cl.numBucketsLog2 = 10;
                int[] C_test_index_array = { 44172927, 72999161, 74326391, 129606274, 21736814 };
                int[] C_test_rank_array = { 61147, 882988, 266290, 133997595, 133525895 };
                cl.testIndexArray = C_test_index_array;
                cl.testRankArray = C_test_rank_array;
                break;
                
            case 'D':
                System.err.println("Warning: Not yet implemented for 1 and 2 worker, over it's ok");
                cl.totalKeysLog2 = 31;
                cl.maxKeyLog2 = 27;
                cl.numBucketsLog2 = 10;
                int[] D_test_index_array = {44172927, 72999161, 74326391, 129606274, 21736814};
                int[] D_test_rank_array = {974930, 14139196, 4271338, 133997595, 133525895};
                cl.testIndexArray = D_test_index_array;
                cl.testRankArray = D_test_rank_array;
        }

        //common variables
        cl.kernelName = "IS";
        cl.problemClassName = clss;
        cl.operationType = "keys ranked";
        cl.numProcs = np;
        cl.testArraySize = 5;
        cl.maxIterations = 10;
        cl.totalKeys = (1L << cl.totalKeysLog2);
        cl.maxKey = (1 << cl.maxKeyLog2);
        cl.numBuckets = (1 << cl.numBucketsLog2);
        cl.numKeys = (int) (cl.totalKeys / cl.numProcs); // warning may go out with D class and few procs
        cl.iterations = cl.maxIterations;
        cl.sizeStr = "" + cl.totalKeys;
        cl.size = cl.totalKeys;
        cl.version = "3.2";

        /** ************************************************************** */

        /* On larger numbers of processors, since the keys are (roughly) */
        /* gaussian distributed, the first and last processor sort keys */
        /* in a large interval, requiring array sizes to be larger. Note */
        /* that for large NUM_PROCS, NUM_KEYS is, however, a small number */
        /** ************************************************************** */
        if (cl.numProcs < 256) { // warning may go out with D class and few procs
            cl.buffersSize = (3 * cl.numKeys) / 2;
        } else {
            cl.buffersSize = (3 * cl.numKeys);
        }

        return cl;
    }
}
