
package data.dataset;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.Serializable;
import java.util.Random;


public class DoubleDataSet implements Serializable {

    private int numSamples;
    private int numFeatures;
    private double[] values;

    public DoubleDataSet(int numSamples, int numFeatures) {
        this.numSamples = numSamples;
        this.numFeatures = numFeatures;
        this.values = new double[numSamples * numFeatures];
    }

    public void populateFromFile(String file) throws FileNotFoundException, IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader("/tmp/samples"))) {
            int sampleId = 0;
            int featureId = 0;
            int featureOffset = 0;

            boolean keepReading = true;
            char[] val = new char[1];
            double mod = 1;
            while (reader.read(val) != -1 && keepReading) {
                switch (val[0]) {
                    case ' ':
                        featureId++;
                        break;
                    case '\n':
                        featureId = 0;
                        sampleId++;
                        if (sampleId >= numSamples) {
                            keepReading = false;
                        }

                        break;
                    case '-':
                        mod = -1;
                    case '+':
                        char[] number = new char[23];
                        reader.read(number);
                        double featureVal = new Double(new String(number)) * mod;
                        if (featureId < numFeatures) {
                            values[featureOffset] = featureVal;
                            featureOffset++;
                        }
                        mod = 1;
                        break;

                }
            }

            /*
             * // Alternative method for reading boolean keepReading = true; StringBuilder value = new StringBuilder();
             * char[] val = new char[1]; while (reader.read(val) != -1 && keepReading) { switch (val[0]) { case ' ': if
             * (featureId < numFeatures) { samplesPart[featurePartOffset] = new Double(value.toString());
             * featurePartOffset++; } featureId++; value = new StringBuilder(); break; case '\n': if (featureId <
             * numFeatures) { samplesPart[featurePartOffset] = new Double(value.toString()); featurePartOffset++; } if
             * (featurePartOffset >= partitionSize) { partId++; if (partId < this.samples.length) { samplesPart =
             * samples[partId]; } featurePartOffset = 0; } featureId = 0; sampleId++; if (sampleId >= numSamples) {
             * keepReading = false; } value = new StringBuilder(); break; default: value.append(val[0]); } }
             */
        }
    }

    public void populateRandom(Double[][] featureBoundaries, long randomSeed) {
        double[] boundariesRange = new double[this.numFeatures];
        double[] baseRange = new double[this.numFeatures];
        for (int featureId = 0; featureId < this.numFeatures; featureId++) {
            boundariesRange[featureId] = featureBoundaries[featureId][1] - featureBoundaries[featureId][0];
            baseRange[featureId] = featureBoundaries[featureId][0];
        }

        Random r = new Random();
        r.setSeed(randomSeed);

        int valueOffset = 0;
        while (valueOffset < values.length) {
            for (int featureId = 0; featureId < this.numFeatures; featureId++) {
                double value = r.nextDouble() * boundariesRange[featureId] + baseRange[featureId];
                values[valueOffset] = value;
                valueOffset++;
            }
        }

    }

    public void print() {
        int valueOffset = 0;
        while (valueOffset < values.length) {
            for (int featureId = 0; featureId < this.numFeatures; featureId++) {
                System.out.print(values[valueOffset++] + "\t");
            }
            System.out.println();
        }
        System.out.println("-------------------------------------------");
    }

    public double getValue(int sampleId, int featureIdx) {
        return values[sampleId * this.numFeatures + featureIdx];
    }

    public int getNumSamples() {
        return this.numSamples;
    }

    public int getFeaturesCount() {
        return this.numFeatures;
    }
}
