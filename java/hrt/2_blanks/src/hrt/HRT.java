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

package hrt;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import hrt.HRTImpl;


public class HRT {

    private static boolean debug;
    private static List<String> modelOutputs = null;


    public static void main(String args[]) throws Exception {

        /*
         * Parameters: - 0: Debug - 1: HRT script location - 2: User - 3: Number of modeller tasks - 4: Output model
         * path - 5: Start date - 6: Duration
         */

        debug = Boolean.parseBoolean(args[0]);
        String scriptPath = args[1];
        String user = args[2];
        Integer numOfTasks = Integer.parseInt(args[3]);
        String outputModelPath = args[4];
        String startDate = args[5];
        String duration = args[6];

        print_header();

        // Parsing models name
        StringTokenizer st = new StringTokenizer(outputModelPath, "/");
        String modelName = null;

        while (st.hasMoreElements()) {
            modelName = st.nextToken();
        }

        if (debug) {
            System.out.println("Parameters: ");
            System.out.println("- Debug Enabled");
            System.out.println("- HRT script: " + scriptPath);
            System.out.println("- User: " + user);
            System.out.println("- Number of modeling tasks: " + numOfTasks);
            System.out.println("- Output model path: " + outputModelPath);
            System.out.println("- Model name: " + modelName);
            System.out.println("- Start date: " + startDate);
            System.out.println("- Duration: " + duration);
            System.out.println(" ");
        }

        Long startTotalTime = System.currentTimeMillis();

        try {
            String lastMerge = "";

            System.out.println("\nCalculating the model:");

            // Creating working directories
            (new File(outputModelPath)).mkdirs();
            (new File(outputModelPath + "monitoring/")).mkdirs();

            // Creating configuration file for modelers
            // TODO: Create confFile (configuration File path) & call genConfigFile to generate it.
            String confFile = "";

            modelOutputs = new ArrayList<String>(numOfTasks);

            // Submitting hrt modeler jobs
            for (int i = 0; i < numOfTasks; i++) {
                modelOutputs.add(outputModelPath + "monitoring/model_" + i + ".log");
                HRTImpl.modeling(scriptPath, confFile, user, i, modelOutputs.get(i));
            }

            // Final monitoring log merge process
            try {
                // Final monitoring log -> Merge 2 by 2
                int neighbor = 1;
                while (neighbor < modelOutputs.size()) {
                    for (int result = 0; result < modelOutputs.size(); result += 2 * neighbor) {
                        if (result + neighbor < modelOutputs.size()) {
                            HRTImpl.mergeMonitorLogs(modelOutputs.get(result), modelOutputs.get(result + neighbor));
                            if (debug)
                                System.out.println(
                                        " - Merging files -> " + modelOutputs.get(result) + " and " + modelOutputs.get(result + neighbor));

                            lastMerge = modelOutputs.get(result);
                        }
                    }
                    neighbor *= 2;
                }
            } catch (Exception e) {
                System.out.println("Error assembling partial results to final result file.");
                e.printStackTrace();
            }

            // Synchronizing last merged monitoring file
            FileInputStream fis = new FileInputStream(lastMerge);
            String monitorOutput = outputModelPath + "monitoring/" + modelName + ".log";

            if (debug)
                System.out.println("\nMoving last merged file: " + lastMerge + " to " + monitorOutput + " \n");

            copyFile(fis, new File(monitorOutput));
            fis.close();

            // Cleaning up partial results
            CleanUp();

            Long stopTotalTime = System.currentTimeMillis();
            Long totalTime = (stopTotalTime - startTotalTime) / 1000;
            System.out.println("\n" + modelName + " computed successfully in " + totalTime + " seconds \n");

        } catch (Exception e) {
            System.out.println("Error: ");
            e.printStackTrace();
        }
    }

    private static void CleanUp() {

        // Cleaning intermediate files
        for (int i = 0; i < modelOutputs.size(); i++) {
            File fres = new File(modelOutputs.get(i));
            fres.delete();
        }
    }

    private static void copyFile(FileInputStream sourceFile, File destFile) throws IOException {
        try (FileChannel source = sourceFile.getChannel();
                FileOutputStream outputDest = new FileOutputStream(destFile);
                FileChannel destination = outputDest.getChannel()) {

            destination.transferFrom(source, 0, source.size());

        } catch (IOException ioe) {
            throw ioe;
        }
    }

    private static void print_header() {
        System.out.println("\nHRT modeling Tool:\n");
    }

}
