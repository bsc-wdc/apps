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
package wordcount.uniqueFile;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Wordcount {

    private static String DATA_FILE;
    private static String OUTPUT_FOLDER;
    private static String OUTPUT_FILE_NAME;
    private static int BLOCK_SIZE;
    private static int NUM_BLOCKS;
    private static double fileLength;

    private static HashMap<String, Integer>[] partialResult;
    private static HashMap<String, Integer> result;

    public static void main(String args[]) {
        // Get parameters
        if (args.length != 4) {
            System.out.println("[ERROR] Usage: Wordcount <DATA_FILE> <OUTPUT_FOLDER> <OUTPUT_FILE_NAME> <BLOCK_SIZE>");
            System.exit(-1);
        }
        DATA_FILE = args[0];
        OUTPUT_FOLDER = args[1];
        OUTPUT_FILE_NAME = args[2];
        BLOCK_SIZE = Integer.valueOf(args[3]);
        
        System.out.println("[PARAMS] DATA_FILE parameter value = " + DATA_FILE);
        System.out.println("[PARAMS] OUTPUT_FOLDER parameter value = " + OUTPUT_FOLDER);
        System.out.println("[PARAMS] OUTPUT_FILE_NAME parameter value = " + OUTPUT_FILE_NAME);
        System.out.println("[PARAMS] BLOCK_SIZE parameter value = " + BLOCK_SIZE);

        // Run wordcount app
        long startTime = System.currentTimeMillis();
        run();
        System.out.println("[LOG] Main program finished.");
        System.out.println("[LOG] Result size = " + result.keySet().size());
        long endTime = System.currentTimeMillis();

        System.out.println("[TIMER] Elapsed time: " + (endTime - startTime) + " ms");
    }

    private static void run() {
        // Initialization
        System.out.println("[LOG] Initialising filenames for each matrix");
        initializeVariables();

        // Map
        System.out.println("[LOG] Performing wordcount");
        int start = 0;
        for (int i = 0; i < NUM_BLOCKS; ++i) {
            partialResult[i] = wordCount(DATA_FILE, start, BLOCK_SIZE);
            start = start + BLOCK_SIZE;
        }
        // MERGE-REDUCE                
        int neighbor = 1;
        while (neighbor < NUM_BLOCKS) {
            for (int result = 0; result < NUM_BLOCKS; result += 2 * neighbor) {
                if (result + neighbor < NUM_BLOCKS) {
                    partialResult[result] = reduceTask(partialResult[result],
                            partialResult[result + neighbor]);
                }
            }
            neighbor *= 2;
        }
        // Sync
        result = partialResult[0];
        int elems = saveAsFile(partialResult[0]);
        // for (int i = 0; i < NUM_BLOCKS; ++i) {
        //     System.out.println("["+i+"] Elems: " + partialResult[i]);//.size());
        // }
    }

    /**
     * Initialize variables.
     */
    private static void initializeVariables() {
        fileLength = (double) (new File(DATA_FILE).length());
        NUM_BLOCKS = (int) Math.ceil(fileLength / ((double) (BLOCK_SIZE)));
        partialResult = (HashMap<String, Integer>[]) new HashMap[NUM_BLOCKS];
        result = new HashMap<>();

        System.out.println("[INIT] fileLength: " + fileLength + " bytes");
        System.out.println("[INIT] NUM_BLOCKS: " + NUM_BLOCKS);
        System.out.println("[INIT] BLOCK_SIZE: " + BLOCK_SIZE + " bytes");
    }

    /**
     * Count words of a file from byte 'start' until 'start' + 'bsize'
     * @param filePath File to read
     * @param start Reading starting point
     * @param bsize Amount of bytes to read
     * @return HashMap<String, Integer> with the accumulated appearances of each word.
     */
    public static HashMap<String, Integer> wordCount(String filePath, int start, int bsize) {
        File file = new File(filePath);
        HashMap<String, Integer> res = new HashMap<String, Integer>();
        FileReader fr = null;
        BufferedReader br = null;
        try {
            RandomAccessFile f = new RandomAccessFile(filePath,"r");
            byte[] buffer = new byte[bsize];
            f.seek(start);
            f.read(buffer);
            
            String line = new String(buffer, StandardCharsets.UTF_8).replace('\n', ' ');
            String[] words = line.trim().split(" ");
            for (String word : words) {
                if (res.containsKey(word)) {
                    res.put(word, res.get(word) + 1);
                } else {
                    res.put(word, 1);
                }
            }
        } catch (Exception e) {
            System.err.println("ERROR: Cannot retrieve values from " + file.getName());
            e.printStackTrace();
        } finally {
            if (br != null) {
                try {
                    br.close();
                } catch (Exception e) {
                    System.err.println("ERROR: Cannot close buffered reader on file " + file.getName());
                    e.printStackTrace();
                }
            }
            if (fr != null) {
                try {
                    fr.close();
                } catch (Exception e) {
                    System.err.println("ERROR: Cannot close file reader on file " + file.getName());
                    e.printStackTrace();
                }
            }
        }

        return res;
    }
    
    /**
     * Join two hashmaps of wordcounts.
     * @param m1 First hashmap
     * @param m2 Second hasmap
     * @return The merging result
     */
    public static HashMap<String, Integer> reduceTask(HashMap<String, Integer> m1, HashMap<String, Integer> m2) {
        for (Iterator<String> iterator = m2.keySet().iterator(); iterator.hasNext();) {
            String key = iterator.next();
            if (m1.containsKey(key)) {
                m1.put(key, (m1.get(key) + m2.get(key)));
            } else {
                m1.put(key, m2.get(key));
            }
        }
        return m1;
    }
    
    /**
     * Write the results in file
     * @param result HashMap<String, Integer> to write
     * @return  The number of words written
     */
    private static int saveAsFile(HashMap<String, Integer> result) {
        // Write to file
        int i = 0;
        File resultsFile = new File(OUTPUT_FOLDER + File.separator + OUTPUT_FILE_NAME);
        try {
            FileOutputStream fos = new FileOutputStream(resultsFile);
            PrintWriter pw = new PrintWriter(fos);
            for (Map.Entry<String, Integer> m : result.entrySet()) {
                pw.println(m.getKey() + "=" + m.getValue());
                i++;
            }
            pw.flush();
            pw.close();
            fos.close();
        } catch (Exception e) {
            System.err.println("ERROR: Cannot save results on file " + resultsFile.getName());
            e.printStackTrace();
        }
        return i;
    }
}
