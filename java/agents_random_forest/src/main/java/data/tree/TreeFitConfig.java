
package data.tree;

import java.io.Serializable;

public class TreeFitConfig implements Serializable {

    final int distrDepth;
    final int numCandidateFeat;
    final int maxDepth;


    public TreeFitConfig(int distrDepth, int numCandidateFeat, int maxDepth) {
        this.distrDepth = distrDepth;
        this.numCandidateFeat = numCandidateFeat;
        this.maxDepth = maxDepth;
    }

    public int getDistributionDepth() {
        return distrDepth;
    }

    public int getNumCandidateFeatures() {
        return numCandidateFeat;
    }

    public int getMaxDepth() {
        return maxDepth;
    }

}
