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


public class Reducer {

    public static ISProblemClass ISProblem;

    public static void accumBucketSizeTotals(int[] bucket_size_totals, int[] bucket_size){
        int arraySize = ISProblem.numBuckets + ISProblem.testArraySize;
        for (int i = 0; i < arraySize; i++) {
            bucket_size_totals[i] += bucket_size[i];
        }
    }

    public static void transferCount(int[][]recv_count, int[] send_count,int rank){
        for (int j=0;j<ISProblem.numProcs;j++){
            recv_count[j][rank]=send_count[j];
        }
    }


}
