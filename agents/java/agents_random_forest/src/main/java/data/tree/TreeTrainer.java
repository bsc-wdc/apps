package data.tree;

import data.dataset.DoubleDataSet;
import data.dataset.IntegerDataSet;


public class TreeTrainer {
    
    public static Tree trainTreeWithDataset(DoubleDataSet samples, IntegerDataSet classification,
        IntegerDataSet selection, TreeFitConfig config, long seed) {
        Tree tree = Tree.trainTreeWithDataset(samples, classification, selection, config, seed);
        return tree;
    }
}
