package simple.files;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;


public class Simple {

    public static final boolean DEBUG = true;

    public static final int DEFAULT_NUM_SIMPLE_APPS = 1;
    public static final int DEFAULT_NUM_INCREMENTS = 1;


    private static void writeValueToFile(String filePath, int value) {
        try (BufferedWriter br = new BufferedWriter(new FileWriter(new File(filePath)))) {
            String valueString = String.valueOf(value) + "\n";
            br.write(String.valueOf(valueString));
        } catch (IOException ioe) {
            System.err.println("ERROR: Exception writing file");
            ioe.printStackTrace();
        }
    }

    private static int readValueFromFile(String filePath) {
        try (BufferedReader br = new BufferedReader(new FileReader(new File(filePath)))) {
            String valueString = br.readLine();
            int val = Integer.valueOf(valueString.trim());
            return val;
        } catch (IOException ioe) {
            System.err.println("ERROR: Exception writing file");
            ioe.printStackTrace();
        }

        return -1;
    }

    // Task
    public static void increment(String filePath) {
        final int initialValue = readValueFromFile(filePath);
        final int incrValue = initialValue + 1;
        writeValueToFile(filePath, incrValue);
    }

    private static void simpleApp(int simpleAppCounter, int numIncrements) {
        // Create file counter
        String filePath = "counter" + simpleAppCounter;
        final int initialValue = 1;
        writeValueToFile(filePath, initialValue);
        if (DEBUG) {
            System.out.println("- Simple app " + simpleAppCounter + " uses file " + filePath);
        }

        // Execute increment task
        for (int i = 0; i < numIncrements; ++i) {
            increment(filePath);
        }

        // Sync file
        final int expectedFinalValue = initialValue + numIncrements;
        final int finalValue = readValueFromFile(filePath);
        if (expectedFinalValue != finalValue) {
            System.err.println("ERROR: Simple app " + simpleAppCounter + " does not fit expected result value ("
                + expectedFinalValue + " != " + finalValue + ")");
        } else {
            if (DEBUG) {
                System.out.println("- Simple app " + simpleAppCounter + " finished with counter = " + finalValue);
            }
        }

    }

    /**
     * Main application that launches numSimpleApps simple applications that increment a counter numIncrements times.
     * 
     * @param numSimpleApps Number of simple applications to launch.
     * @param numIncrements Number of counter increments performed by each application.
     */
    public static void executeSimpleApps(int numSimpleApps, int numIncrements) {
        // Run simple apps
        if (DEBUG) {
            System.out.println("Run simple apps");
        }
        for (int i = 1; i <= numSimpleApps; ++i) {
            if (DEBUG) {
                System.out.println("Running simple " + i + "/" + numSimpleApps);
            }
            simpleApp(i, numIncrements);
        }
        if (DEBUG) {
            System.out.println("DONE");
        }
    }

    /**
     * Entry point.
     * 
     * @param args System arguments.
     */
    public static void main(String[] args) {
        final int numSimpleApps = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_NUM_SIMPLE_APPS;
        final int numIncrements = (args.length >= 1) ? Integer.parseInt(args[0]) : DEFAULT_NUM_INCREMENTS;

        executeSimpleApps(numSimpleApps, numIncrements);
    }
}
