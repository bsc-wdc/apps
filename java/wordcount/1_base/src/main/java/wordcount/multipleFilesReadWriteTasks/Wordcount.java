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
package wordcount.multipleFilesReadWriteTasks;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.Map;

public class Wordcount {

    private static String DATA_FOLDER;
    private static String OUTPUT_FOLDER;
    private static String OUTPUT_FILE_NAME;

    private static String[] filePaths;
    private static HashMap<String, Integer>[] partialResult;
    private static int result;

    public static void main(String args[]) {
        // Get parameters
        if (args.length != 3) {
            System.out.println("[ERROR] Usage: Wordcount <DATA_FOLDER> <OUTPUT_FOLDER> <OUTPUT_FILE_NAME>");
            System.exit(-1);
        }
        DATA_FOLDER = args[0];
        OUTPUT_FOLDER = args[1];
        OUTPUT_FILE_NAME = args[2];

        System.out.println("DATA_FOLDER parameter value = " + DATA_FOLDER);
        System.out.println("OUTPUT_FOLDER parameter value = " + OUTPUT_FOLDER);
        System.out.println("OUTPUT_FILE_NAME parameter value = " + OUTPUT_FILE_NAME);
        
        // Run wordcount app
        long startTime = System.currentTimeMillis();
        run();
        System.out.println("[LOG] Main program finished.");
        System.out.println("[LOG] Result words = " + result);
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
            partialResult[i] = wordCount(read(fp));
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
        result = write(partialResult[0]);
    }

    /**
     * Initialize variables.
     */
    private static void initializeVariables() {
        int numFiles = new File(DATA_FOLDER).listFiles().length;
        filePaths = new String[numFiles];
        int i = 0;
        for (File f : new File(DATA_FOLDER).listFiles()) {
            filePaths[i] = f.getAbsolutePath();
            i = i + 1;
        }

        partialResult = (HashMap<String, Integer>[]) new HashMap[numFiles];
        result = 0;
    }

    /**
     * Read a file and return its content as arraylist of strings.
     * Each entry corresponds to a file line.
     * @param filePath File to read
     * @return ArrayList<String> with the contents of the file.
     */
    public static ArrayList<String> read(String filePath) {
        File file = new File(filePath);
        FileReader fr = null;
        BufferedReader br = null;
        ArrayList<String> content = new ArrayList<String>();
        try {
            fr = new FileReader(file);
            br = new BufferedReader(fr);
            String line;
            while ((line = br.readLine()) != null) {
                content.add(line);
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
        return content;
    }

    /**
     * Count words of an ArrayList<String>
     * @param content ArrayList<String> of words to count
     * @return HashMap<String, Integer> with the accumulated appearances of each word.
     */
    public static HashMap<String, Integer> wordCount(ArrayList<String> content) {
        HashMap<String, Integer> res = new HashMap<String, Integer>();
        for (String line : content) {
            String[] words = line.split(" ");
            for (String word : words) {
                if (res.containsKey(word)) {
                    res.put(word, res.get(word) + 1);
                } else {
                    res.put(word, 1);
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
     * @return The number of words written
     */
    public static int write(HashMap<String, Integer> result) {
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

