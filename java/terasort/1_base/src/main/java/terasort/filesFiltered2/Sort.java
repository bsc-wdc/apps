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
package terasort.filesFiltered2;

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
        if (args.length != 3) {
            System.out.println("[ERROR] Usage: Sort <DATA_FOLDER> <OUTPUT_FOLDER> <NUM_RANGES>");
            System.exit(-1);
        }
        DATA_FOLDER = args[0];
        System.out.println("[LOG] DATA_FOLDER   = " + DATA_FOLDER);
        OUTPUT_FOLDER = args[1];
        System.out.println("[LOG] OUTPUT_FOLDER = " + OUTPUT_FOLDER);
        NUM_RANGES = Integer.parseInt(args[2]);
        System.out.println("[LOG] NUM_RANGES = " + NUM_RANGES);

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

    /*
	 * This run reads only the range that requires from each file.
	 * Avoids to transfer the entire input file.
     */
    private void run() {
        // Initialize file Names
        System.out.println("[LOG] Initialising filenames for each matrix");
        initializeVariables();

        // Compute result
        System.out.println("[LOG] Reading fragments.");
        System.out.println("[LOG] Filtering fragments.");

        Fragment[][] fragments = new Fragment[NUM_FRAGMENTS][NUM_RANGES];
        Range[][] ranges = new Range[NUM_FRAGMENTS][NUM_RANGES];
        long step = ((maxKey - minKey) / NUM_RANGES);

        for (int i = 0; i < NUM_FRAGMENTS; ++i) {
            long start = 0;
            long end = step;
            for (int j = 0; j < NUM_RANGES; ++j) {
                fragments[i][j] = SortImpl.getFilteredFragment(filePaths[i], start, end);
                ranges[i][j] = new Range(start, end);
                start += step;
                end += step;
            }
        }

        System.out.println("[LOG] Merging filtered results...");
        System.out.println("[LOG] Sorting fragment...");
        System.out.println("[LOG] Saving sorted fragments...");

        int nf = NUM_FRAGMENTS;
        int nr = NUM_RANGES;
        Range[][] aux;
        Fragment[][] auxFragments;
        while (nf > 1) {
            nf = nf / 2;
            nr = nr * 2;
            aux = new Range[nf][nr];
            auxFragments = new Fragment[nf][nr];
            for (int i = 0; i < nf; i++) {
                for (int j = 0; j < nr; j++) {
                    auxFragments[i][j] = new Fragment();
                }
            }
            int posX = 0;
            for (int i = 0; i < nf * 2; i += 2) {
                int posY = 0;
                for (int j = 0; j < nr / 2; j++) {
                    long s = ranges[i][j].getStart();
                    long e = ranges[i + 1][j].getEnd();
                    Range r = new Range(s, e);
                    SortImpl.mixFragments(fragments[i][j], fragments[i + 1][j], r, auxFragments[posX][posY], auxFragments[posX][posY + 1]);
                    long h = s + ((e - s) / 2);
                    aux[posX][posY] = new Range(s, h);
                    aux[posX][posY + 1] = new Range(h, e);
                    posY += 2;
                }
                posX++;
            }
            ranges = aux;
            fragments = auxFragments;
        }
        Integer[] sortedCount = new Integer[fragments[0].length];

        // sort all fragments
        for (int i = 0; i < fragments[0].length; i++) {
            Fragment sortedFragment = SortImpl.sortPartition(fragments[0][i]);
            sortedCount[i] = SortImpl.saveFragment(sortedFragment, OUTPUT_FOLDER);
        }

        // Merge-Reduce counters
        LinkedList<Integer> q = new LinkedList<Integer>();
        for (int j = 0; j < fragments[0].length; ++j) {
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

        System.out.println("[TERASORT] FINISHED");
    }

    private void initializeVariables() {
        NUM_FRAGMENTS = new File(DATA_FOLDER).listFiles().length;
        filePaths = new String[NUM_FRAGMENTS];
        int i = 0;
        for (File f : new File(DATA_FOLDER).listFiles()) {
            filePaths[i] = f.getAbsolutePath();
            System.out.println("File: " + filePaths[i]);
            i = i + 1;
        }

        //result = new Fragment[NUM_FRAGMENTS];
        byte[] mi = {0, 0, 0, 0, 0, 0, 0, 0};
        minKey = Utils.byte2long(mi);
        byte[] ma = {0, -1, -1, -1, -1, -1, -1, -1}; // 0xff = -1
        maxKey = Utils.byte2long(ma);
        System.out.println("Min key: " + minKey);
        System.out.println("Max key: " + maxKey);
    }

}
