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


public class Verifier {

    public static ISProblemClass ISProblem;

    public static int full_verify2(int k, int rank, int[] total_local_keys, int[] key_array) {
        int i;
        int j = 0;
        if (rank!=0) {
            if (k > key_array[0]) {
                System.out.println("Boundary element incorrect on proc " + rank + "; k, key = " + k +", " + key_array[0]);
                j++;
            }
        }

        /* Confirm keys correctly sorted: count incorrectly sorted keys, if any */
        for (i = 1; i < total_local_keys[0]; i++)
            if (key_array[i - 1] > key_array[i]) {
                System.out.println("Internal element incorrect on proc " + rank + "; i, km1, k = " + i +
                    ", " + key_array[i - 1] + ", " + key_array[i]);
                j++;
            }

        if (j != 0) {
            System.out.println("Processor " + rank + ":  Full_verify: number of keys out of sort: " + j);
            return 0;
        } else {
            return 1;
        }
    }

    public static int getK(int[] key_array, int[] total_local_key){
        return key_array[total_local_key[0] - 1];
    }

    public static boolean parcial_verify( int[]verifies) {
        return (verifies[0]>0);
    }

    public static boolean final_verify( int passed_verification) {
        return passed_verification!=0;
        
    }

    public static void checkValues(String string, int[] i) {
        int checksum=0;
        for (int k=0;k<i.length;k++)
        {
            checksum+=i[k];
        }
        System.out.println(string + checksum);
    }
}
