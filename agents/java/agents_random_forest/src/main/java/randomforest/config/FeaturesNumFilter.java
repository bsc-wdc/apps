package randomforest.config;

public class FeaturesNumFilter {

    public static enum Config {
        NONE, SQRT, THIRD
    }


    /**
     * Computes the number of features to consider when looking for the best split.
     *
     * @param numFeatures dataset's number of features
     * @param config configuration filter
     * @return number of features
     */
    public static int resolveNumCandidateFeatures(int numFeatures, Config config) {
        int numberFeatures = 0;
        switch (config) {
            case SQRT:
                numberFeatures = (int) Math.sqrt(numFeatures);
                break;
            case THIRD:
                numberFeatures = Math.max(1, numFeatures / 3);
                break;
            default:
                numberFeatures = numFeatures;
        }
        return numberFeatures;
    }
}
