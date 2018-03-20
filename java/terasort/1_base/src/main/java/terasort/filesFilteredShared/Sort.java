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
package terasort.filesFilteredShared;

import java.io.File;

public class Sort {

    private static int NUM_FILES;
    private static int NUM_BUCKETS;
    private static int NUM_BLOCKS;
    private static long minKey;
    private static long maxKey;
    private static String DATA_FOLDER;
    private static String OUTPUT_FOLDER;
    private static String TEMP_FOLDER;
    private static boolean REMOVE_RESULTS;
    private static File[] BUCKETS;
    private static String[] BUCKETS_PATH;
    private static String[] filePaths;

    public static void main(String args[]) {
        // Get parameters
        if (args.length != 5) {
            System.out.println("[ERROR] Usage: Sort <DATA_FOLDER> <OUTPUT_FOLDER> <NUM_BLOCKS> <NUM_BUCKETS> <REMOVE_RESULTS>");
            System.exit(-1);
        }
        DATA_FOLDER = args[0];
        System.out.println("[LOG] DATA_FOLDER   = " + DATA_FOLDER);
        OUTPUT_FOLDER = args[1];
        System.out.println("[LOG] OUTPUT_FOLDER = " + OUTPUT_FOLDER);
        NUM_BLOCKS = Integer.parseInt(args[2]);
        setupOutputFolder();
        System.out.println("[LOG] NUM_BLOCKS = " + NUM_BLOCKS);
        NUM_BUCKETS = Integer.parseInt(args[3]);
        System.out.println("[LOG] NUM_BUCKETS = " + NUM_BUCKETS);
        REMOVE_RESULTS = Boolean.parseBoolean(args[4]);
        System.out.println("[LOG] REMOVE_RESULTS = " + REMOVE_RESULTS);

        // Run sort by key app
        long startTime = System.currentTimeMillis();
        Sort sbk = new Sort();
        sbk.run();
        System.out.println("[LOG] Main program finished.");
        long endTime = System.currentTimeMillis();
        System.out.println("[TIMER] Elapsed time: " + (endTime - startTime) + " ms");
        if (REMOVE_RESULTS) {
            SortImpl.deleteFolder(new File(OUTPUT_FOLDER));
        }
    }

    /**
     * Execution funcion.
     */
    private void run() {
        // Initialize file Names and variables
        System.out.println("[LOG] Initialising filenames for each matrix");
        initializeVariables();
        setupTempFolder();
        System.out.println("[LOG] Reading fragments.");
        System.out.println("[LOG] Filtering fragments.");
        int totalBlocks = NUM_FILES * NUM_BLOCKS;
        Integer[] ranges = new Integer[totalBlocks];
        long bucketStep = ((maxKey - minKey) / NUM_BUCKETS);
        long firstFileSize = new File(filePaths[0]).length();
        long blockStep = firstFileSize / NUM_BLOCKS;

        // MAP all ranges per fragment
        int part = 0;
        int blockNumber = 0;
        for (int i = 0; i < NUM_FILES; ++i) {
            long blockPos = 0;
            for (int j = 0; j < NUM_BLOCKS; ++j) {
                ranges[blockNumber] = SortImpl.getFilteredFragment(filePaths[i], blockPos, blockPos + blockStep, BUCKETS_PATH, bucketStep, part);
                blockPos += blockStep;
                part += NUM_BUCKETS;
                blockNumber++;
            }
        }
        System.out.println("[LOG] Merging filtered resulfts...");
        System.out.println("[LOG] Sorting fragment...");
        System.out.println("[LOG] Saving sorted fragments...");
        Long[] sortedCount = new Long[NUM_BUCKETS];

        // Merge-Reduce per reader
        int neighbor = 1;
        while (neighbor < totalBlocks) {
            for (int i = 0; i < totalBlocks; i += 2 * neighbor) {
                if (i + neighbor < totalBlocks) {
                    ranges[i] = SortImpl.reduceBuckets(ranges[i], ranges[i + neighbor]);
                }
            }
            neighbor *= 2;
        }
        // Map all sorts and save
        for (int j = 0; j < NUM_BUCKETS; j++) {
            sortedCount[j] = SortImpl.sortBucket(BUCKETS_PATH[j], OUTPUT_FOLDER + "/part_" + j, ranges[0]);
        }
        // Merge-Reduce counters
        neighbor = 1;
        while (neighbor < NUM_BUCKETS) {
            for (int i = 0; i < NUM_BUCKETS; i += 2 * neighbor) {
                if (i + neighbor < NUM_BUCKETS) {
                    sortedCount[i] = SortImpl.reduceSortedBuckets(sortedCount[i], sortedCount[i + neighbor]);
                }
            }
            neighbor *= 2;
        }
        long count = sortedCount[0];
        System.out.println("[TERASORT] Total Sorted Elements = " + count);
        System.out.println("[TERASORT] Cleaning temporary folder.");
        SortImpl.deleteFolder(new File(TEMP_FOLDER));
        System.out.println("[TERASORT] FINISHED");
    }

    /**
     * Initialize variables.
     */
    private void initializeVariables() {
        int files = 0;
        for (File f : new File(DATA_FOLDER).listFiles()) {
            if (f.isFile()) {
                files++;
            }
        }
        NUM_FILES = files;
        filePaths = new String[NUM_FILES];
        int i = 0;
        for (File f : new File(DATA_FOLDER).listFiles()) {
            if (f.isFile()) {
                filePaths[i] = f.getAbsolutePath();
                System.out.println("[LOG] File: " + filePaths[i]);
                i = i + 1;
            }
        }
        byte[] mi = {0, 0, 0, 0, 0, 0, 0, 0};
        minKey = Utils.byte2long(mi);
        byte[] ma = {0, -1, -1, -1, -1, -1, -1, -1}; // 0xff = -1
        maxKey = Utils.byte2long(ma);
        System.out.println("[LOG] Min key: " + minKey);
        System.out.println("[LOG] Max key: " + maxKey);
        BUCKETS = new File[NUM_BUCKETS];
        BUCKETS_PATH = new String[NUM_BUCKETS];
    }

    /**
     * Create the output folder.
     */
    public static void setupOutputFolder() {
        File outputDir = new File(OUTPUT_FOLDER);
        if (outputDir.exists()) {
            SortImpl.deleteFolder(outputDir);
        }
        SortImpl.createFolder(outputDir);
    }

    /**
     * Create the temporary bucket folders.
     */
    public static void setupTempFolder() {
        TEMP_FOLDER = OUTPUT_FOLDER + "/temp";
        File tempDir = new File(TEMP_FOLDER);
        if (tempDir.exists()) {
            SortImpl.deleteFolder(tempDir);
        }
        SortImpl.createFolder(tempDir);
        System.out.println("[LOG] Creating temporary bucket folders");
        for (int i = 0; i < NUM_BUCKETS; i++) {
            BUCKETS[i] = new File(TEMP_FOLDER + "/" + i);
            SortImpl.createFolder(BUCKETS[i]);
            BUCKETS_PATH[i] = BUCKETS[i].getAbsolutePath();
        }
    }
}
