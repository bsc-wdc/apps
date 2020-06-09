
package randomforest;

import randomforest.config.DataSetConfig;
import data.dataset.DoubleDataSet;
import data.dataset.IntegerDataSet;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;


/**
 * A class to encapsulate an input set of Points for use in the RandomForest program and the reading/writing of a set of
 * points to data files.
 */
public class RandomForestDataSet {

    private final int numSamples;
    private final int numFeatures;

    private DoubleDataSet samples;
    private IntegerDataSet classes;

    public RandomForestDataSet(int numSamples, int numFeatures) {
        this.numSamples = numSamples;
        this.numFeatures = numFeatures;

        this.samples = new DoubleDataSet(numSamples, numFeatures);
        this.classes = new IntegerDataSet(numSamples, 1);

    }

    public void populateRandom(DataSetConfig config) throws IOException, InterruptedException {
        Long randomSeed = config.getRandomSeed();
        if (randomSeed == null) {
            randomSeed = new Random().nextLong();
        }
        String pythonProgram = "from sklearn.datasets import make_classification \n" + "import numpy as np \n"
                + "samples, classes = make_classification(" + "n_samples=" + config.getNumSamples() + ", " + "n_features="
                + config.getNumFeatures() + ", " + "n_classes=" + config.getNumClasses() + ", " + "n_informative="
                + config.getNumInformative() + ", " + "n_redundant=" + config.getNumRedundant() + ", " + "n_repeated="
                + config.getNumRepeated() + ", " + "n_clusters_per_class=" + config.getNumClustersPerClass() + ", "
                + "shuffle=" + (config.isShuffle() ? "True" : "False") + ", " + "random_state=" + randomSeed + ")\n"
                + "with open('/tmp/samples', 'wb') as f:\n" + "    np.savetxt(fname=f, X=samples, fmt='%+.17e') \n"
                + "with open('/tmp/classes', 'wb') as f:\n" + "    np.savetxt(fname=f, X=classes.astype(int), fmt='%i')";

        final Process p = Runtime.getRuntime().exec(new String[]{"python",
            "-c",
            pythonProgram});
        p.getOutputStream().close();
        p.waitFor();
        p.getInputStream().close();

        BufferedReader br = new BufferedReader(new InputStreamReader(p.getErrorStream()));
        String line;
        while ((line = br.readLine()) != null) {
            System.err.println(line);
        }
        p.getErrorStream().close();

        this.samples.populateFromFile("/tmp/samples");
        this.classes.populateFromFile("/tmp/classes");
    }

    public int getNumSamples() {
        return numSamples;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public IntegerDataSet getClasses() {
        return classes;
    }

    public DoubleDataSet getSamples() {
        return samples;
    }
}
