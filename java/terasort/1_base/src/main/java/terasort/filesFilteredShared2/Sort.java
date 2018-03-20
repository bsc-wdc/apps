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
package terasort.filesFilteredShared2;

import java.io.File;

public class Sort {

    private static int NUM_FILES;
    private static int NUM_BUCKETS;
    private static int NUM_BLOCKS;
    private static long minKey;
    private static long maxKey;
    private static String DATA_FOLDER;
    private static String OUTPUT_FOLDER;
    private static boolean REMOVE_RESULTS;

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
        // Initialize file Names
        System.out.println("[LOG] Initialising filenames for each matrix");
        initializeVariables();

        // Compute result
        System.out.println("[LOG] Reading fragments.");
        System.out.println("[LOG] Filtering fragments.");

        int totalBlocks = NUM_FILES * NUM_BLOCKS;
        long bucketStep = ((maxKey - minKey) / NUM_BUCKETS);
        long firstFileSize = new File(filePaths[0]).length();
        long blockStep = firstFileSize / NUM_BLOCKS;

        Fragment[] fragments = new Fragment[totalBlocks];
        Fragment[][] buckets = new Fragment[totalBlocks][NUM_BUCKETS];

        // Map - Read all blocks
        int blockNumber = 0;
        for (int i = 0; i < NUM_FILES; ++i) {
            long blockPos = 0;
            for (int j = 0; j < NUM_BLOCKS; ++j) {
                fragments[blockNumber] = SortImpl.readBlock(filePaths[i], blockPos, blockPos + blockStep);
                long bucketPos = 0;
                for (int k = 0; k < NUM_BUCKETS; ++k) {
                    buckets[blockNumber][k] = SortImpl.extractSortedBucket(fragments[blockNumber], bucketPos, bucketPos + bucketStep);
                    bucketPos += bucketStep;
                }
                blockPos += blockStep;
                blockNumber++;
            }
        }

        // One merge-reduce per bucket
        Long[] elems = new Long[NUM_BUCKETS];
        for (int k = 0; k < NUM_BUCKETS; ++k) {
            int neighbor = 1;
            while (neighbor < totalBlocks) {
                for (int i = 0; i < totalBlocks; i += 2 * neighbor) {
                    if (i + neighbor < totalBlocks) {
                        buckets[i][k] = SortImpl.mergeBuckets(buckets[i][k], buckets[i + neighbor][k]);
                    }
                }
                neighbor *= 2;
            }
            elems[k] = SortImpl.saveFragment(buckets[0][k], OUTPUT_FOLDER + "/part_" + k);
        }

        // Merge-Reduce counters
        int neighbor = 1;
        while (neighbor < NUM_BUCKETS) {
            for (int i = 0; i < NUM_BUCKETS; i += 2 * neighbor) {
                if (i + neighbor < NUM_BUCKETS) {
                    elems[i] = SortImpl.reduceSortedBucketsCount(elems[i], elems[i + neighbor]);
                }
            }
            neighbor *= 2;
        }
        long count = elems[0];

        System.out.println("[TERASORT] Total Sorted Elements = " + count);
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
}
