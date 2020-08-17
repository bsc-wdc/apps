package data.tree;

import data.dataset.DoubleDataSet;
import data.dataset.IntegerDataSet;
import data.utils.ComparatorValuedPair;
import data.utils.ValuedPair;

import java.io.Externalizable;
import java.io.IOException;
import java.io.ObjectInput;
import java.io.ObjectOutput;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Objects;
import java.util.Random;
import java.util.TreeSet;


public class Tree implements Externalizable {

    private Node root;


    public Tree() {
        // Only for externalisation
    }

    public static Tree trainTreeWithDataset(DoubleDataSet samples, IntegerDataSet classification,
        IntegerDataSet selection, TreeFitConfig config, long seed) {
        Tree tree = new Tree();
        final LinkedList<Object[]> treeTraversal = new LinkedList<>();
        final Random random = new Random();
        random.setSeed(seed);
        treeTraversal.add(new Object[] { tree,
            selection,
            0 });

        while (!treeTraversal.isEmpty()) {
            Object[] iterationVals = treeTraversal.pop();
            Tree currentTree = (Tree) iterationVals[0];
            IntegerDataSet subSelection = (IntegerDataSet) iterationVals[1];
            int depth = (int) iterationVals[2];
            // if (depth < config.getDistributionDepth()) {
            long traversalSeed = random.nextLong();
            Object[] splitResult = splitNode(samples, classification, subSelection, config, traversalSeed);
            if (splitResult[0] != null && depth < config.getMaxDepth()) {
                int featureId = (int) splitResult[0];
                double splitValue = (double) splitResult[1];
                IntegerDataSet lowerSelection = (IntegerDataSet) splitResult[2];
                IntegerDataSet upperSelection = (IntegerDataSet) splitResult[3];

                Tree lowerChild = new Tree();
                Tree upperChild = new Tree();

                currentTree.setSplitRoot(featureId, splitValue, lowerChild, upperChild);

                treeTraversal.add(new Object[] { lowerChild,
                    lowerSelection,
                    depth + 1 });
                treeTraversal.add(new Object[] { upperChild,
                    upperSelection,
                    depth + 1 });

            } else {
                int[] totalClasses = (int[]) splitResult[4];
                currentTree.setJointRoot(totalClasses);
            }
        }

        return tree;
    }

    private static Object[] splitNode(DoubleDataSet samples, IntegerDataSet classification, IntegerDataSet selection,
        TreeFitConfig config, long seed) {
        final Random r = new Random();
        r.setSeed(seed);
        LinkedList<Integer> untriedFeatureIdxs = new LinkedList<>();
        for (int featureId = 0; featureId < samples.getFeaturesCount(); featureId++) {
            untriedFeatureIdxs.add(featureId);
        }

        IntegerDataSet lowerChild = new IntegerDataSet(0, 1);
        IntegerDataSet upperChild = new IntegerDataSet(0, 1);

        int bestIndex;
        double bestValue;
        int[] totalClassesCount = new int[0];

        while (untriedFeatureIdxs.size() > 0) {
            long iterationRandomSeed = r.nextLong();
            int[] featureIdxSelection =
                popRandomFeatures(untriedFeatureIdxs, config.getNumCandidateFeatures(), iterationRandomSeed);

            double bestScore = Double.MAX_VALUE;
            bestIndex = -1;
            bestValue = -1;
            for (int index : featureIdxSelection) {
                Object[] splitResult = testSplit(samples, classification, selection, index);
                totalClassesCount = (int[]) splitResult[4];
                if (splitResult[0] != null) {
                    double score = (double) splitResult[1];
                    if (score < bestScore) {
                        bestScore = score;
                        bestIndex = index;
                        bestValue = (double) splitResult[0];
                        lowerChild = (IntegerDataSet) splitResult[2];
                        upperChild = (IntegerDataSet) splitResult[3];
                    }
                }
            }
            if (bestIndex >= 0) {
                // Get groups
                return new Object[] { bestIndex,
                    bestValue,
                    lowerChild,
                    upperChild,
                    totalClassesCount };
            }
        }
        return new Object[] { null,
            null,
            null,
            null,
            totalClassesCount };
    }

    private static int[] popRandomFeatures(LinkedList<Integer> untriedFeatureIdxs, int numCandidateFeatures,
        long seed) {
        Random random = new Random();
        random.setSeed(seed);
        int maxPositions = Math.min(numCandidateFeatures, untriedFeatureIdxs.size());
        int[] result = new int[maxPositions];
        for (int i = 0; i < maxPositions; i++) {
            int idx = random.nextInt(untriedFeatureIdxs.size());
            int featureIdx = untriedFeatureIdxs.remove(idx);
            result[i] = featureIdx;
        }
        return result;
    }

    private static Object[] testSplit(DoubleDataSet samples, IntegerDataSet classification, IntegerDataSet selection,
        int featureIdx) {
        int totalNumSamples = selection.getNumSamples();
        if (totalNumSamples == 0) {
            return new Object[] { null,
                null,
                null,
                null,
                new int[0] };
        }

        // Sort the selection according to the feature (featureIdx) value
        TreeSet<ValuedPair> sortedSelection = new TreeSet<>(new ComparatorValuedPair());

        int numClasses = classification.getMaxValuePerFeature(0) + 1;
        int[] totalClassesCount = new int[numClasses];
        for (int sampleId : selection.getValues()) {
            double val = ((DoubleDataSet) samples).getValue(sampleId, featureIdx);
            int classIdx = ((IntegerDataSet) classification).getValue(sampleId, 0);
            totalClassesCount[classIdx]++;
            sortedSelection.add(new ValuedPair(sampleId, val, classIdx));
        }

        double bestValue = -Double.MAX_VALUE;
        Double lastValue = sortedSelection.first().getValue();
        int lastClassId = sortedSelection.first().getClassId();
        int partialSamples = 0;
        int[] partialClassesCount = new int[numClasses];
        double bestScore = Double.MAX_VALUE;

        ArrayList<Integer> underThreshold = new ArrayList<>();
        LinkedList<Integer> upperThreshold = new LinkedList<>();

        // Evaluating score for all values of the partition
        for (ValuedPair vp : sortedSelection) {
            if (!Objects.equals(lastValue, vp.getValue()) && lastClassId != vp.getClassId()) {
                double cutValue = (lastValue + vp.getValue()) / 2;
                double score =
                    evaluateSplitDifference(partialSamples, totalNumSamples, partialClassesCount, totalClassesCount);

                if (score < bestScore) {
                    bestScore = score;
                    bestValue = cutValue;
                    underThreshold.addAll(upperThreshold);
                    upperThreshold.clear();
                }
            }

            partialClassesCount[vp.getClassId()]++;
            partialSamples++;
            lastValue = vp.getValue();
            lastClassId = vp.getClassId();
            upperThreshold.add(vp.getId());
        }

        if (bestScore == Double.MAX_VALUE) {
            return new Object[] { null,
                null,
                null,
                null,
                totalClassesCount };
        }
        // Generate corresponding subSelections
        IntegerDataSet underSelection = new IntegerDataSet(underThreshold.size(), 1);
        IntegerDataSet upperSelection = new IntegerDataSet(upperThreshold.size(), 1);
        upperSelection.populateFromList(upperThreshold);
        underSelection.populateFromList(underThreshold);
        return new Object[] { bestValue,
            bestScore,
            underSelection,
            upperSelection,
            totalClassesCount };
    }

    private static double evaluateSplitDifference(int partialPosition, int totalSize, int[] partialClassesCount,
        int[] totalClassesCount) {
        double aboveScore = 0;
        double underScore = 0;
        for (int classIdx = 0; classIdx < totalClassesCount.length; classIdx++) {
            aboveScore += partialClassesCount[classIdx] * partialClassesCount[classIdx];
            double underOccurences = totalClassesCount[classIdx] - partialClassesCount[classIdx];
            underScore += underOccurences * underOccurences;
        }
        return -(aboveScore / (double) partialPosition + underScore / (double) (totalSize - partialPosition));
    }

    private void setSplitRoot(int featureId, double splitValue, Tree lowerChild, Tree upperChild) {
        this.root = new SplitNode(featureId, splitValue, lowerChild, upperChild);
    }

    private void setJointRoot(int[] frequency) {
        this.root = new LeafNode(frequency);
    }

    public void print(String firstPrefix, String nextPrefix) {
        if (this.root != null) {
            this.root.print(firstPrefix, nextPrefix);
        } else {
            System.out.println(firstPrefix);
        }
    }

    @Override
    public void writeExternal(ObjectOutput oo) throws IOException {
        oo.writeObject(this.root);
    }

    @Override
    public void readExternal(ObjectInput oi) throws IOException, ClassNotFoundException {
        this.root = (Node) oi.readObject();
    }

}
