
package randomforest.config;

import data.tree.TreeFitConfig;


public class FitConfig {

    private int numEstimators;
    private int distrDepth;
    private int numCandidateFeat;
    private int maxDepth;
    private Long randomSeed;


    public FitConfig(int numEstimators, int distrDepth, int numCandidateFeat, int maxDepth, Long randomSeed) {
        this.numEstimators = numEstimators;
        this.distrDepth = distrDepth;
        this.numCandidateFeat = numCandidateFeat;
        this.maxDepth = maxDepth;
        this.randomSeed = randomSeed;
    }

    public TreeFitConfig getTreeFitConfig() {
        return new TreeFitConfig(this.distrDepth, this.numCandidateFeat, this.maxDepth);
    }

    public int getNumEstimators() {
        return this.numEstimators;
    }

    public Long getRandomSeed() {
        return this.randomSeed;
    }

    public void print() {
        System.out.println("Applying random forest with parameters:");
        System.out.println("\t* numEstimators: " + this.numEstimators);
        System.out.println("\t* number of candidate features: " + this.numCandidateFeat);
        System.out.println("\t* maxDepth: " + this.maxDepth);
        System.out.println("\t* distrDepth: " + this.distrDepth);
        System.out.println("\t* randomState: " + this.randomSeed);
    }
}
