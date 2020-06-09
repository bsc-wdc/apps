
package randomforest.config;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;


public class DataSetConfig implements Externalizable {

    private int numSamples;
    private int numFeatures;
    private int numClasses;

    // Generation Parameters
    private int numInformative;
    private int numRedundant;
    private int numClustersPerClass;
    private int numRepeated;
    private boolean shuffle;
    private Long randomSeed;


    public DataSetConfig() {
    }

    public DataSetConfig(int numSamples, int numFeatures, int numClasses, int numInformative, int numRedundant,
        int numClustersPerClass, int numRepeated, boolean shuffle, Long randomSeed) {
        this.numSamples = numSamples;
        this.numFeatures = numFeatures;
        this.numClasses = numClasses;
        this.numInformative = numInformative;
        this.numRedundant = numRedundant;
        this.numClustersPerClass = numClustersPerClass;
        this.numRepeated = numRepeated;
        this.shuffle = shuffle;
        this.randomSeed = randomSeed;
    }

    public void print() {
        System.out.println("Generating random forest dataset with values:");

        System.out.println("\t* randomState: " + randomSeed);
    }

    public int getNumSamples() {
        return numSamples;
    }

    public int getNumFeatures() {
        return numFeatures;
    }

    public Long getRandomSeed() {
        return randomSeed;

    }

    public int getNumClasses() {
        return numClasses;
    }

    public int getNumInformative() {
        return numInformative;
    }

    public int getNumRedundant() {
        return numRedundant;
    }

    public int getNumClustersPerClass() {
        return numClustersPerClass;
    }

    public int getNumRepeated() {
        return numRepeated;
    }

    public boolean isShuffle() {
        return shuffle;
    }

    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeInt(numSamples);
        oo.writeInt(numFeatures);
        oo.writeInt(numClasses);
        oo.writeInt(numInformative);
        oo.writeInt(numRedundant);
        oo.writeInt(numClustersPerClass);
        oo.writeInt(numRepeated);
        oo.writeBoolean(shuffle);
        oo.writeObject(randomSeed);
    }

    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        numClasses = oi.readInt();
        numSamples = oi.readInt();
        numFeatures = oi.readInt();
        numInformative = oi.readInt();
        numRedundant = oi.readInt();
        numClustersPerClass = oi.readInt();
        numRepeated = oi.readInt();
        shuffle = oi.readBoolean();
        randomSeed = (Long) oi.readObject();
    }

}
