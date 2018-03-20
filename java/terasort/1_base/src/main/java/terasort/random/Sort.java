/*
 *  Copyright 2002-2016 Barcelona Supercomputing Center (www.bsc.es)
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
package terasort.random;

import java.util.LinkedList;

public class Sort {

    private static int NUM_KEYS;
    private static int UNIQUE_KEYS;
    private static int KEY_LENGTH;
    private static int UNIQUE_VALUES;
    private static int VALUE_LENGTH;
    private static int NUM_FRAGMENTS;
    private static int RANDOM_SEED;
    private static int KEYS_PER_FRAGMENT;
    private static int NUM_RANGES;

    private static Fragment[] result;

    public static void main(String args[]) {
        //Get parameters
        if (args.length != 7) {
            System.out.println("[ERROR] Usage: Sort <NUM_KEYS> <UNIQUE_KEYS> <KEY_LENGTH>");
            System.out.println("                    <UNIQUE_VALUES> <VALUE_LENGTH> <NUM_FRAGMENTS> <RANDOM_SEED>");
            System.exit(-1);
        }
        NUM_KEYS = Integer.valueOf(args[0]);
        UNIQUE_KEYS = Integer.valueOf(args[1]);
        KEY_LENGTH = Integer.valueOf(args[2]);
        UNIQUE_VALUES = Integer.valueOf(args[3]);
        VALUE_LENGTH = Integer.valueOf(args[4]);
        NUM_FRAGMENTS = Integer.valueOf(args[5]);
        RANDOM_SEED = Integer.valueOf(args[6]);
        KEYS_PER_FRAGMENT = NUM_KEYS / NUM_FRAGMENTS;
        NUM_RANGES = NUM_FRAGMENTS;

        System.out.println("NUM_KEYS parameter value = " + NUM_KEYS);
        System.out.println("UNIQUE_KEYS parameter value = " + UNIQUE_KEYS);
        System.out.println("KEY_LENGTH parameter value = " + KEY_LENGTH);
        System.out.println("UNIQUE_VALUES parameter value = " + UNIQUE_VALUES);
        System.out.println("VALUE_LENGTH parameter value = " + VALUE_LENGTH);
        System.out.println("NUM_FRAGMENTS parameter value = " + NUM_FRAGMENTS);
        System.out.println("RANDOM_SEED parameter value = " + RANDOM_SEED);
        System.out.println("KEYS_PER_FRAGMENT parameter value = " + KEYS_PER_FRAGMENT);
        System.out.println("NUM_RANGES parameter value = " + NUM_RANGES);

        // Run sort by key app
        long startTime = System.currentTimeMillis();
        Sort sbk = new Sort();
        sbk.run();

        System.out.println("[LOG] Main program finished.");

        // Syncronize result
        // System.out.println("[LOG] Result size = " + result.size());
        // Uncomment the following lines to see the result keys
        for (int i = 0; i < NUM_RANGES; ++i) {
            System.out.println("Range " + i + " result:");
            System.out.println(result[i]);
        }
        long endTime = System.currentTimeMillis();

        System.out.println("[TIMER] Elapsed time: " + (endTime - startTime) + " ms");
    }

    /**
     * Execution method.
     */
    private void run() {
        // Initialize file Names
        System.out.println("[LOG] Initialising filenames for each matrix");
        initializeVariables();

        // Compute result
        System.out.println("[LOG] Computing result");
        Fragment[][] ranges = new Fragment[NUM_FRAGMENTS][NUM_RANGES];
        for (int i = 0; i < NUM_FRAGMENTS; ++i) {
            Fragment fragment = SortImpl.generateFragment(KEYS_PER_FRAGMENT, UNIQUE_KEYS, KEY_LENGTH, UNIQUE_VALUES, VALUE_LENGTH, RANDOM_SEED + (long) 2 * KEYS_PER_FRAGMENT * i);
            int start = 0;
            int step = UNIQUE_KEYS / NUM_RANGES;
            int end = step;
            for (int j = 0; j < NUM_RANGES; ++j) {
                ranges[i][j] = SortImpl.filterTask(fragment, start, end);
                start += step;
                end += step;
            }
        }
        for (int i = 0; i < NUM_RANGES; ++i) {
            // MERGE-REDUCE
            LinkedList<Integer> q = new LinkedList<Integer>();
            for (int j = 0; j < NUM_FRAGMENTS; ++j) {
                q.add(j);
            }
            int x = 0;
            while (!q.isEmpty()) {
                x = q.poll();
                int y;
                if (!q.isEmpty()) {
                    y = q.poll();
                    ranges[x][i] = SortImpl.reduceTask(ranges[x][i], ranges[y][i]);
                    q.add(x);
                }
            }

            result[i] = SortImpl.sortPartition(ranges[x][i]);
        }
        int sortedCount = 0;
        for (int i = 0; i < NUM_RANGES; ++i) {
            sortedCount += result[i].getCount();
        }
        System.out.println("Total Sorted Elements = " + sortedCount);
    }

    /**
     * Variables initialization.
     */
    private void initializeVariables() {
        result = new Fragment[NUM_FRAGMENTS];
    }

}
