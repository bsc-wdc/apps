package multi;

import data.dataset.DoubleDataSet;
import data.dataset.IntegerDataSet;
import data.tree.Tree;
import data.tree.TreeFitConfig;

import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.task.Method;


public interface MultiRandomForestItf {

    @Method(declaringClass = "randomforest.RandomForest")
    IntegerDataSet randomSelection(
        @Parameter(direction = Direction.IN) int lowerBoundary,
        @Parameter(direction = Direction.IN) int upperBoundary,
        @Parameter(direction = Direction.IN) int numElements,
        @Parameter(direction = Direction.IN) long randomSeed
    );

    @Method(declaringClass = "data.tree.TreeTrainer")
    Tree trainTreeWithDataset(
        @Parameter(direction = Direction.IN) DoubleDataSet samples,
        @Parameter(direction = Direction.IN) IntegerDataSet classification,
        @Parameter(direction = Direction.IN) IntegerDataSet selection,
        @Parameter(direction = Direction.IN) TreeFitConfig config,
        @Parameter(direction = Direction.IN) long seed
    );
    
    @Method(declaringClass = "randomforest.RandomForest")
    void generateRandomModelWithTest(
        @Parameter() int numSamples, 
        @Parameter() int numFeatures, 
        @Parameter() int numClasses, 
        @Parameter() int numInformative,
        @Parameter() int numRedundant, 
        @Parameter() int numRepeated, 
        @Parameter() int numClustersPerClass, 
        @Parameter() boolean shuffle, 
        @Parameter() long randomState,
        @Parameter() int numEstimators, 
        @Parameter() int numModels
    );

}
