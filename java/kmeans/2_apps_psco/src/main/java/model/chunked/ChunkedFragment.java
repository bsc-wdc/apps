package model.chunked;

import java.util.ArrayList;
import java.util.Random;

import model.Clusters;
import model.SumPoints;
import serialization.DataClayObject;


/**
 * A class to encapsulate an input set of Points for use in the KMeans program and the reading/writing of a set of
 * points to data files.
 */
@SuppressWarnings("serial")
public class ChunkedFragment extends DataClayObject {

    private ArrayList<Chunk> chunks;
    private int maxChunkSize;


    public ChunkedFragment(int newMaxChunkSize) {
        chunks = new ArrayList<Chunk>();
        maxChunkSize = newMaxChunkSize;
    }

    public ChunkedFragment(int foo, final String alias) {
        super(alias);
    }

    public ChunkedFragment(float[][] points, int maxChunkSize) {
        for (int i = 0; i < points.length; i++) {
            addVector(points[i]);
        }
    }

    public void fillPoints(int idForSeed, int newSizeFrag, int newNDimensions) {
        Random random = new Random(idForSeed);
        for (int i = 0; i < newSizeFrag; i++) {
            float[] vector = new float[newNDimensions];
            for (int j = 0; j < newNDimensions; j++) {
                vector[j] = random.nextFloat() * 2 - 1;
            }
            addVector(vector);
        }
    }

    private void addVector(float[] f) {
        Chunk curChunk = null;
        boolean createNewChunk = false;
        int curSize = chunks.size();
        if (curSize == 0) {
            createNewChunk = true;
        } else {
            curChunk = chunks.get(curSize - 1);
            if (curChunk.getSize() == maxChunkSize) {
                createNewChunk = true;
            }
        }
        if (createNewChunk) {
            curChunk = new Chunk();
            curChunk.newProxy(true, this.getLocation());
        }
        curChunk.addVector(f);
    }

    // @task
    public Clusters clusters_points_partial(ChunkedFragment mu, int k, int ind) {
        Clusters clustersOfFrag = new Clusters(k);

        int numDimensions = this.getNumDimensions();
        for (int p = 0; p < this.getNumVectors(); p++) {
            int closest = -1;
            float closestDist = Float.MAX_VALUE;
            for (int m = 0; m < mu.getNumVectors(); m++) {
                float dist = 0;
                for (int dim = 0; dim < numDimensions; dim++) {
                    float tmp = this.getPoint(p, dim) - mu.getPoint(m, dim);
                    dist += tmp * tmp;
                }
                if (dist < closestDist) {
                    closestDist = dist;
                    closest = m; // cluster al que pertenece
                }
            }
            int value = ind + p;
            clustersOfFrag.addIndex(closest, value);
        }
        // StorageLocationID location = this.getLocation();
        clustersOfFrag.makePersistent(true, null);

        return clustersOfFrag;
    }

    // @task
    public SumPoints partial_sum(Clusters cluster, int k, int ind) {
        SumPoints pSum = new SumPoints(k, this.getNumDimensions());
        // cluster.getSize = k
        for (int c = 0; c < cluster.getSize(); c++) {
            int[] positionValues = cluster.getIndexes(c);
            for (int i = 0; i < cluster.getIndexesSize(c); i++) {
                int value = positionValues[i];
                float[] v = this.getVector(value - ind);
                pSum.sumValue(c, v, 1);
            }
        }
        pSum.makePersistent(true, null);
        return pSum;
    }

    public float getPoint(int v, int dim) {
        int chunk;
        int pos;
        if (v > maxChunkSize) {
            chunk = v / maxChunkSize;
            pos = v - (maxChunkSize * chunk);
        } else {
            chunk = 0;
            pos = v;
        }
        return chunks.get(chunk).getVector(pos)[dim];
    }

    public float[] getVector(int v) {
        int chunk;
        int pos;
        if (v > maxChunkSize) {
            chunk = v / maxChunkSize;
            pos = v - (maxChunkSize * chunk);
        } else {
            chunk = 0;
            pos = v;
        }
        return chunks.get(chunk).getVector(pos);
    }

    public int getNumVectors() {
        int totalSize = 0;
        for (Chunk c : chunks) {
            totalSize += c.getSize();
        }
        return totalSize;
    }

    public int getNumDimensions() {
        return chunks.get(0).getDimensionsPerVector();
    }

    public void updatePoints(float[][] pts) {
        int curVector = 0;
        for (Chunk c : chunks) {
            for (int i = 0; i < maxChunkSize && curVector < pts.length; i++) {
                c.updateVector(i, pts[curVector]);
                curVector++;
            }
            if (curVector == pts.length) {
                break;
            }
        }
    }
    
}
