package data.tree;

import java.io.Serializable;


public class TreeFitConfig implements Serializable {

    private static final long serialVersionUID = 1L;

    private final int distrDepth;
    private final int numCandidateFeat;
    private final int maxDepth;


    public TreeFitConfig(int distrDepth, int numCandidateFeat, int maxDepth) {
        this.distrDepth = distrDepth;
        this.numCandidateFeat = numCandidateFeat;
        this.maxDepth = maxDepth;
    }

    public int getDistributionDepth() {
        return this.distrDepth;
    }

    public int getNumCandidateFeatures() {
        return this.numCandidateFeat;
    }

    public int getMaxDepth() {
        return this.maxDepth;
    }

}
