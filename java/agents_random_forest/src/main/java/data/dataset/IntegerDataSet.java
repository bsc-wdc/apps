
package data.dataset;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Random;


public class IntegerDataSet implements Serializable {

    private int numSamples;
    private int numFeatures;
    private int[] values;

    private int[] maxValue;
    private int[] minValue;

    public IntegerDataSet() {
        this(0, 0);
    }

    public IntegerDataSet(int numSamples, int numFeatures) {
        this.numSamples = numSamples;
        this.numFeatures = numFeatures;
        this.values = new int[numSamples * numFeatures];
        this.maxValue = new int[numFeatures];
        this.minValue = new int[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            maxValue[i] = Integer.MIN_VALUE;
            minValue[i] = Integer.MAX_VALUE;
        }
    }

    public void populateFromList(ArrayList<Integer> list) {
        int sampleId = 0;
        for (int val : list) {
            if (sampleId >= numSamples) {
                break;
            }

            values[sampleId] = val;
            if (minValue[0] > val) {
                minValue[0] = val;
            }
            if (maxValue[0] < val) {
                maxValue[0] = val;
            }
            sampleId++;
        }
    }

    public void populateFromList(LinkedList<Integer> list) {
        int sampleId = 0;
        for (int val : list) {
            if (sampleId >= numSamples) {
                break;
            }

            values[sampleId] = val;
            if (minValue[0] > val) {
                minValue[0] = val;
            }
            if (maxValue[0] < val) {
                maxValue[0] = val;
            }
            sampleId++;
        }
    }

    public void populateFromFile(String file) throws FileNotFoundException, IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader("/tmp/classes"))) {
            int sampleId = 0;
            String line;
            while (sampleId < numSamples && ((line = reader.readLine()) != null)) {
                int value = new Integer(line);
                values[sampleId] = value;
                if (minValue[0] > value) {
                    minValue[0] = value;
                }
                if (maxValue[0] < value) {
                    maxValue[0] = value;
                }
                sampleId++;
            }
        }
    }

    public void populateRandom(Integer[][] featureBoundaries, long randomSeed) {

        int[] boundariesRange = new int[this.numFeatures];
        int[] baseRange = new int[this.numFeatures];
        for (int featureId = 0; featureId < this.numFeatures; featureId++) {
            boundariesRange[featureId] = featureBoundaries[featureId][1] - featureBoundaries[featureId][0];
            baseRange[featureId] = featureBoundaries[featureId][0];
        }

        Random r = new Random();
        r.setSeed(randomSeed);

        int valueOffset = 0;
        while (valueOffset < values.length) {
            for (int featureId = 0; featureId < this.numFeatures; featureId++) {
                int value = (int) (r.nextDouble() * (double) boundariesRange[featureId]) + baseRange[featureId];
                values[valueOffset] = value;
                if (minValue[featureId] > value) {
                    minValue[featureId] = value;
                }
                if (maxValue[featureId] < value) {
                    maxValue[featureId] = value;
                }
                valueOffset++;
            }
        }
    }

    public void print() {
        System.out.println("-------------------------------------------");
        int valueOffset = 0;
        while (valueOffset < values.length) {
            for (int featureId = 0; featureId < this.numFeatures; featureId++) {
                System.out.print(values[valueOffset++]);
            }
            System.out.println();
        }
        System.out.println("-------------------------------------------");

    }

    public int getMaxValuePerFeature(int featureId) {
        return this.maxValue[featureId];
    }

    public int getMinValuePerFeature(int featureId) {
        return this.minValue[featureId];
    }

    public int[] getValues() {
        return values;
    }

    public int getValue(int sampleId, int featureIdx) {
        return values[sampleId * this.numFeatures + featureIdx];
    }

    public int getNumSamples() {
        return this.numSamples;
    }

    public int getFeaturesCount() {
        return this.numFeatures;
    }
}
