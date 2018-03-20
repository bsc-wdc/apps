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
package terasort.files;

import java.io.File;
import java.util.LinkedList;

public class Sort {

    private static int NUM_FRAGMENTS;
    private static int NUM_RANGES;
    private static long minKey;
    private static long maxKey;
    private static String DATA_FOLDER;
    private static String OUTPUT_FOLDER;

    private static String[] filePaths;
    //private static Fragment[] result;

    public static void main(String args[]) {
        // Get parameters
        if (args.length != 2) {
            System.out.println("[ERROR] Usage: Sort <DATA_FOLDER> <OUTPUT_FOLDER>");
            System.exit(-1);
        }
        DATA_FOLDER = args[0];
        System.out.println("[LOG] DATA_FOLDER   = " + DATA_FOLDER);
        OUTPUT_FOLDER = args[1];
        System.out.println("[LOG] OUTPUT_FOLDER = " + OUTPUT_FOLDER);

        // Run sort by key app
        long startTime = System.currentTimeMillis();
        Sort sbk = new Sort();
        sbk.run();

        System.out.println("[LOG] Main program finished.");

        // Uncomment the following lines to see the result keys
        //for (int i=0; i < NUM_RANGES; ++i){
        //	System.out.println("Range " + i + " result:");
        //	System.out.println(result[i]);
        //}
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
        System.out.println("[LOG] Reading fragments.");
        System.out.println("[LOG] Filtering fragments.");
        Fragment[][] ranges = new Fragment[NUM_FRAGMENTS][NUM_RANGES];
        for (int i = 0; i < NUM_FRAGMENTS; ++i) {
            Fragment fragment = SortImpl.getFragment(filePaths[i]);
            long start = 0;
            long step = ((maxKey - minKey) / NUM_RANGES);
            long end = step;
            for (int j = 0; j < NUM_RANGES; ++j) {
                ranges[i][j] = SortImpl.filterTask(fragment, start, end);
                start += step;
                end += step;
            }
        }
        System.out.println("[LOG] Merging filtered results.");
        System.out.println("[LOG] Sorting fragment.");
        System.out.println("[LOG] Saving sorted fragments.");
        Integer[] sortedCount = new Integer[NUM_RANGES];
        for (int i = 0; i < NUM_RANGES; ++i) {
            // Merge-Reduce per range
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
            Fragment result = SortImpl.sortPartition(ranges[x][i]);
            sortedCount[i] = SortImpl.saveFragment(result, OUTPUT_FOLDER);
        }

        // Merge-Reduce counters
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
                sortedCount[x] = SortImpl.reduceCount(sortedCount[x], sortedCount[y]);
                q.add(x);
            }
        }
        int count = sortedCount[x];

        System.out.println("[TERASORT] Total Sorted Elements = " + count);
    }

    /**
     * Variables initialization.
     */
    private void initializeVariables() {
        NUM_FRAGMENTS = new File(DATA_FOLDER).listFiles().length;
        filePaths = new String[NUM_FRAGMENTS];
        int i = 0;
        for (File f : new File(DATA_FOLDER).listFiles()) {
            filePaths[i] = f.getAbsolutePath();
            System.out.println("File: " + filePaths[i]);
            i = i + 1;
        }
        System.out.println("NUM_FRAGMENTS value = " + NUM_FRAGMENTS);

        //result = new Fragment[NUM_FRAGMENTS];
        NUM_RANGES = NUM_FRAGMENTS;

        byte[] mi = {0, 0, 0, 0, 0, 0, 0, 0};
        minKey = Utils.byte2long(mi);
        byte[] ma = {0, -1, -1, -1, -1, -1, -1, -1}; // 0xff = -1
        maxKey = Utils.byte2long(ma);
        System.out.println("Min key: " + minKey);
        System.out.println("Max key: " + maxKey);
    }

}
