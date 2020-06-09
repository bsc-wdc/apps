package kmeans;

import java.io.DataOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Random;


/**
 * A class to encapsulate an input set of Points for use in the KMeans program and the reading/writing of a set of
 * points to data files.
 */
public class KMeansDataSet {

    private static final int cookie = 0x2badfdc0;
    private static final int version = 1;

    public final int numPoints;
    public final int numDimensions;
    public final float[][] points;
    public final float[] currentCluster;


    public KMeansDataSet(int np, int nd, float[][] pts, float[] cluster) {
        assert np * nd == pts.length;
        numPoints = np;
        numDimensions = nd;
        points = pts;
        currentCluster = cluster;
    }

    /*
     * public final float getFloat(int point, int dim) { return points[point*numDimensions + dim]; }
     */

    public final int getPointOffset(int point) {
        return point * numDimensions;
    }

    /**
     * Create numPoints random points each of dimension numDimensions.
     */
    public static KMeansDataSet generateRandomPoints(int numPoints, int numDimensions, int numFrags, int K) {
        float[][] points = new float[numFrags][];
        float[] cluster = new float[K * numDimensions];

        return new KMeansDataSet(numPoints, numDimensions, points, cluster);
    }

    /**
     * Generate a set of random points and write them to a data file
     * 
     * @param fileName the name of the file to create
     * @param numPoints the number of points to write to the file
     * @param numDimensions the number of dimensions each point should have
     * @param seed a random number seed to generate the points.
     * @return <code>true</code> on success, <code>false</code> on failure
     */
    public static boolean generateRandomPointsToFile(String fileName, int numPoints, int numDimensions, int seed) {
        try {
            Random rand = new Random(seed);
            File outputFile = new File(fileName);
            DataOutputStream out = new DataOutputStream(new FileOutputStream(outputFile));
            out.writeInt(cookie);
            out.writeInt(version);
            out.writeInt(numPoints);
            out.writeInt(numDimensions);
            int numFloats = numPoints * numDimensions;
            for (int i = 0; i < numFloats; i++) {
                out.writeFloat(rand.nextFloat());
            }
            out.close();
        } catch (FileNotFoundException e) {
            System.out.println("Unable to open file for writing " + fileName);
            return false;
        } catch (IOException e) {
            System.out.println("Error writing data to " + fileName);
            e.printStackTrace();
            return false;
        }

        return true;
    }
}
