package consumer;

import model.Fragment;
import model.SumPoints;

public class KMeansImpl {
    
    // Task RETURN IN IN IN IN
    public static SumPoints clusters_points_and_partial_sum(Fragment fragment, Fragment mu, int k, int ind) {
        return fragment.cluster_points_and_partial_sum(mu, k, ind);
    }

    // TASK RETURN IN IN
    public static SumPoints reduceCentersTask(SumPoints a, SumPoints b) {
        return a.reduceCentersTask(b);
    }

}
