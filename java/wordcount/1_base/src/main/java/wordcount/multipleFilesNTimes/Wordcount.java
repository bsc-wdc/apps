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
package wordcount.multipleFilesNTimes;


import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Wordcount {

    private static String DATA_FOLDER;
    private static String OUTPUT_FOLDER;
    private static String OUTPUT_FILE_NAME;
    private static int N_TIMES;
    
    private static String[] filePaths;
    private static HashMap<String, Integer>[] partialResult;
    private static HashMap<String, Integer> result;

    public static void main(String args[]) {
        // Get parameters
        if (args.length != 4) {
            System.out.println("[ERROR] Usage: Wordcount <DATA_FOLDER> <OUTPUT_FOLDER> <OUTPUT_FILE_NAME> <N_TIMES>");
            System.exit(-1);
        }
        DATA_FOLDER = args[0];
        OUTPUT_FOLDER = args[1];
        OUTPUT_FILE_NAME = args[2];
        N_TIMES = Integer.parseInt(args[3]);
        System.out.println("DATA_FOLDER parameter value   = " + DATA_FOLDER);
        System.out.println("OUTPUT_FOLDER parameter value = " + OUTPUT_FOLDER);
        System.out.println("OUTPUT_FILE_NAME parameter value = " + OUTPUT_FILE_NAME);
        System.out.println("N_TIMES parameter value = " + N_TIMES);
        // Run wordcount app
        long startTime = System.currentTimeMillis();
        run();
        System.out.println("[LOG] Main program finished.");
        System.out.println("[LOG] Result words = " + result.keySet().size());
        long endTime = System.currentTimeMillis();
        System.out.println("[TIMER] Elapsed time: " + (endTime - startTime) + " ms");
    }

    private static void run() {
        // Initialization
        System.out.println("[LOG] Initialising filenames for each matrix");
        initializeVariables();
        // Map
        System.out.println("[LOG] Performing wordcount");
        int l = filePaths.length;
        for (int i = 0; i < l; ++i) {
            String fp = filePaths[i];
            partialResult[i] = wordCount(fp);
        }
        // MERGE-REDUCE                
        int neighbor = 1;
        while (neighbor < l) {
            for (int result = 0; result < l; result += 2 * neighbor) {
                if (result + neighbor < l) {
                    partialResult[result] = reduceTask(partialResult[result], partialResult[result + neighbor]);
                }
            }
            neighbor *= 2;
        }
        // Sync
        result = partialResult[0];
        int elems = saveAsFile(partialResult[0]);
    }

    /**
     * Initialize variables.
     */
    private static void initializeVariables() {
        int numFiles = new File(DATA_FOLDER).listFiles().length;
        filePaths = new String[numFiles * N_TIMES];
        int i = 0;
        for (int n=0; n<N_TIMES; n++){
            for (File f : new File(DATA_FOLDER).listFiles()) {
                filePaths[i] = f.getAbsolutePath();
                i += 1;
            }
        }
        partialResult = (HashMap<String, Integer>[]) new HashMap[numFiles * N_TIMES];
        result = new HashMap<String, Integer>();
    }

    /**
     * Count words of a single file.
     * @param filePath File to read
     * @return HashMap<String, Integer> with the accumulated appearances of each word.
     */
    public static HashMap<String, Integer> wordCount(String filePath) {
        File file = new File(filePath);
        HashMap<String, Integer> res = new HashMap<String, Integer>();
        FileReader fr = null;
        BufferedReader br = null;
        try {
            fr = new FileReader(file);
            br = new BufferedReader(fr);
            String line;
            while ((line = br.readLine()) != null) {
                String[] words = line.split(" ");
                for (String word : words) {
                    if (res.containsKey(word)) {
                        res.put(word, res.get(word) + 1);
                    } else {
                        res.put(word, 1);
                    }
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
