/*
 *  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package kmeans;

import java.util.Random;


public class KMeans {

	/**
	 * Given the current cluster, compute a new cluster
	 */
	public static void computeNewLocalClusters(int myK, int numDimensions,
			float[] points, float[] clusterPoints, float[] newClusterPoints,
			int[] clusterCounts) {

		int numPoints = points.length / numDimensions;
		for (int pointNumber = 0; pointNumber < numPoints; pointNumber++) {
			int closest = -1;
			float closestDist = Float.MAX_VALUE;
			for (int k = 0; k < myK; k++) {
				float dist = 0;
				for (int dim = 0; dim < numDimensions; dim++) {
					float tmp = points[pointNumber * numDimensions + dim]
							- clusterPoints[k * numDimensions + dim];
					dist += tmp * tmp;
				}
				if (dist < closestDist) {
					closestDist = dist;
					closest = k;
				}
			}

			for (int dim = 0; dim < numDimensions; dim++) {
				newClusterPoints[closest * numDimensions + dim] += points[pointNumber
						* numDimensions + dim];
			}
			clusterCounts[closest]++;
		}
	}

	public static void accumulate(float[] onePoints, float[] otherPoints,
			int[] oneCounts, int[] otherCounts) {
		for (int i = 0; i < otherPoints.length; i++) {
			onePoints[i] += otherPoints[i];
		}
		for (int i = 0; i < otherCounts.length; i++) {
			oneCounts[i] += otherCounts[i];
		}
	}

	public static float[] initPointsFrag(int size, int seed) {
		float[] points = new float[size];
		Random rnd = new Random(seed);
		for (int j = 0; j < size; j++)
			points[j] = rnd.nextFloat();

		return points;
	}

	private static void localReduction(float[] points, int[] counts, int K,
			int numDimensions, float[] cluster) {
		for (int k = 0; k < K; k++) {
			float tmp = (float) counts[k];
			for (int dim = 0; dim < numDimensions; dim++) {
				points[k * numDimensions + dim] /= tmp;
			}
		}

		System.arraycopy(points, 0, cluster, 0, cluster.length);
	}

	private static void initializePoints(KMeansDataSet data, int numFrags) {
		int pointsPerFragment = data.numPoints / numFrags;
		for (int i = 0; i < numFrags; i++) {
			int start = i * pointsPerFragment;
			int stop = Math.min(start + pointsPerFragment - 1, data.numPoints
					* data.numDimensions);
			int numPointsFrag = stop - start + 1;
			data.points[i] = initPointsFrag(numPointsFrag * data.numDimensions,
					i);
		}

		// Initialize cluster (copy first points)
		int nFrag = 0, startPos = 0;
		int toCopy = data.currentCluster.length;
		while (toCopy > 0) {
			int copied = copyToCluster(data.points[nFrag], data.currentCluster,
					toCopy, startPos);
			toCopy -= copied;
			startPos += copied;
			nFrag++;
		}
	}

	private static int copyToCluster(float[] points, float[] cluster,
			int toCopy, int startPos) {
		int canCopy = Math.min(toCopy, points.length);

		int j = 0;
		for (int i = startPos; i < startPos + canCopy; i++)
			cluster[i] = points[j++];

		return j;
	}

	public static void printPoints(float[] points) {
		System.out.println("No print");
		// for (int i = 0; i < points.length; i++)
		// System.out.print(points[i] + " ");
		System.out.println("");
	}

	public static void main(final String[] args) {
		int K = 4;
		int iterations = 50;
		int nPoints = 2000;
		int nDimensions = 2;
		int nFrags = 2;
		//String fileName = "points.dat";
		int argIndex = 0;

		while (argIndex < args.length) {
			String arg = args[argIndex++];
			if (arg.equals("-c")) {
				K = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-i")) {
				iterations = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-n")) {
				nPoints = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-d")) {
				nDimensions = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-f")) {
				nFrags = Integer.parseInt(args[argIndex++]);
				// ///////
			} else {
				//fileName = arg;
			}
		}

		System.out.println("Running with the following parameters:");
		System.out.println("- Clusters: " + K);
		System.out.println("- Points: " + nPoints);
		System.out.println("- Iterations: " + iterations);
		System.out.println("- Dimensions: " + nDimensions);
		System.out.println("- Nodes: " + nFrags);

		// KMeansDataSet data = KMeansDataSet.readPointsFromFile(fileName);
		KMeansDataSet data = KMeansDataSet.generateRandomPoints(nPoints,
				nDimensions, nFrags, K);

		int[][] clusterCounts = new int[nFrags][K];
		float[][] newClusters = new float[nFrags][K * data.numDimensions];

		initializePoints(data, nFrags);

		// Do the requested number of iterations
		for (int iter = 0; iter < iterations; iter++) {
			//long total_iter_time = System.currentTimeMillis();

			// Computation
			for (int i = 0; i < nFrags; i++) {
				float[] frag = data.points[i];
				computeNewLocalClusters(K, nDimensions, frag,
						data.currentCluster, newClusters[i], clusterCounts[i]);
			}

			// Reduction: points and counts
			// Stored in newClusters[0], clusterCounts[0]
			int size = newClusters.length;
			int i = 0, gap = 1;
			while (size > 1) {
				accumulate(newClusters[i], newClusters[i + gap],
						clusterCounts[i], clusterCounts[i + gap]);
				size--;
				i = i + 2 * gap;
				if (i == newClusters.length) {
					gap *= 2;
					i = 0;
				}
			}

			// Local reduction to get the new clusters
			// Adjust cluster coordinates by dividing each point value
			// by the number of points in the cluster
			localReduction(newClusters[0], clusterCounts[0], K,
					data.numDimensions, data.currentCluster);

		}

		// All done. Print the results
		System.out.println("Result clusters: ");
		for (int k = 0; k < K; k++) {
			for (int j = 0; j < data.numDimensions; j++) {
				if (j > 0)
					System.out.print(" ");
				System.out
						.print(data.currentCluster[k * data.numDimensions + j]);
			}
			System.out.println();
		}
		System.out.println();
	}

}
