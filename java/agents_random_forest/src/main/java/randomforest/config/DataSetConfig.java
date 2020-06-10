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
        return this.numSamples;
    }

    public int getNumFeatures() {
        return this.numFeatures;
    }

    public Long getRandomSeed() {
        return this.randomSeed;

    }

    public int getNumClasses() {
        return this.numClasses;
    }

    public int getNumInformative() {
        return this.numInformative;
    }

    public int getNumRedundant() {
        return this.numRedundant;
    }

    public int getNumClustersPerClass() {
        return this.numClustersPerClass;
    }

    public int getNumRepeated() {
        return this.numRepeated;
    }

    public boolean isShuffle() {
        return this.shuffle;
    }

    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeInt(this.numSamples);
        oo.writeInt(this.numFeatures);
        oo.writeInt(this.numClasses);
        oo.writeInt(this.numInformative);
        oo.writeInt(this.numRedundant);
        oo.writeInt(this.numClustersPerClass);
        oo.writeInt(this.numRepeated);
        oo.writeBoolean(this.shuffle);
        oo.writeObject(this.randomSeed);
    }

    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        this.numClasses = oi.readInt();
        this.numSamples = oi.readInt();
        this.numFeatures = oi.readInt();
        this.numInformative = oi.readInt();
        this.numRedundant = oi.readInt();
        this.numClustersPerClass = oi.readInt();
        this.numRepeated = oi.readInt();
        this.shuffle = oi.readBoolean();
        this.randomSeed = (Long) oi.readObject();
    }

}
