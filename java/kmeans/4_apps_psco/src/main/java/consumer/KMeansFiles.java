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
package consumer;

import java.io.File;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;

import model.Clusters;
import model.Fragment;
import model.SumPoints;
import randomness.Randomness;

// UNCOMMENT THIS IF YOU UNCOMMENT COMPSs.waitForAllTasks(); statements
 import es.bsc.compss.api.COMPSs;

public class KMeansFiles {
	static int K = 4; // num of clusters
	static double epsilon = 1e-4; // convergence criteria
	static int iterations = 50; // max iterations to converge

	static String fragmentsDirPath = ""; // folder with fragment files

	static int vectorsPerFragment = 2; // max vectors per fragment
	static int dimsPerVector = 2; // max dimensions per vector

	static boolean doDebug = false;
	static boolean preRead = false;
	static boolean passPaths = false;

	static Fragment muResult; // final result
	static int iterationsDone; // final num of iterations

	static Map<String, Fragment> preCreatedFragments = null;

	public static void main(String[] args) {
		System.out.println("[LOG] Executing KMeans with fragments in files");

		checkArguments(args);

		System.out.println("[LOG] Running with the following parameters:");
		System.out.println("- Fragments dir : " + fragmentsDirPath);
		System.out.println("- Clusters      : " + K);
		System.out.println("- Iterations    : " + iterations);
		System.out.println("- VectorsPerFrag: " + vectorsPerFragment);
		System.out.println("- Dimensions    : " + dimsPerVector);

		// Get fragments file paths
		String[] fragmentsPaths = null;
		File path = new File(fragmentsDirPath);
		File[] files = path.listFiles();
		fragmentsPaths = new String[files.length];
		for (int i = 0; i < files.length; i++) {
			fragmentsPaths[i] = files[i].getPath();
		}
		Utils.sortFragmentPaths(fragmentsPaths);

		// First centroids
		muResult = new Fragment(K, dimsPerVector); // random
		muResult.fillPoints(Randomness.muSeed);
		System.out.println("[LOG] Random mu, 1st vector:" + Arrays.toString(muResult.getVector(0)));

		if (preRead) {
			preCreatedFragments = new HashMap<String, Fragment>();
			for (int i = 0; i < fragmentsPaths.length; i++) {
				Fragment f = new Fragment();
				f.readFromFile(fragmentsPaths[i], vectorsPerFragment, dimsPerVector);
				if (doDebug) {
					System.out.println("[LOG] Read fragment " + i + " from file " + fragmentsPaths[i] + ". 1st vector "
							+ Arrays.toString(f.getVector(0)));
				}
				preCreatedFragments.put(fragmentsPaths[i], f);
			}
			System.out.println("[LOG] Pre-read " + preCreatedFragments.size() + " fragments");
		}

		// KMeans execution
		System.out.println("[LOG] Init computeKMeans");
		long startTime = System.currentTimeMillis();
		computeKMeans(fragmentsPaths);
		 COMPSs.waitForAllTasks();
		long estimatedTime = System.currentTimeMillis() - startTime;
		System.out.println("[TIMER] Elapsed time: " + estimatedTime);

		System.out.println("[RESULT] Iterated " + iterationsDone + " times to get mu:");
		for (int i = 0; i < K; i++) {
			float[] iVector = muResult.getVector(i);
			System.out.println("    " + Arrays.toString(iVector));
		}
	}

	private static void computeKMeans(String[] fragmentsPaths) {
		long init = 0, end;
		int nFrags = fragmentsPaths.length;
		Map<String, Fragment> dataSet = null;
		if (preRead) {
			dataSet = preCreatedFragments;
		} else if (!passPaths) {
			dataSet = new HashMap<String, Fragment>();
			for (int i = 0; i < fragmentsPaths.length; i++) {
				Fragment f = new Fragment();
				f.readFromFile(fragmentsPaths[i], vectorsPerFragment, dimsPerVector);
				if (doDebug) {
					System.out.println("[LOG] Read fragment " + i + " from file " + fragmentsPaths[i] + ". 1st vector "
							+ Arrays.toString(f.getVector(0)));
				}
				dataSet.put(fragmentsPaths[i], f);
			}
			if (doDebug) {
				System.out.println("[LOG] Read " + dataSet.size() + " fragments");
			}
		}

		Fragment oldmu = null;
		int currentIteration = 1;
		// Convergence condition
		while (currentIteration <= iterations && !Utils.has_converged(muResult, oldmu, epsilon)) {
			System.out.println("[LOG] Starting iteration " + currentIteration + " of " + iterations);

			oldmu = muResult;
			// Clusters[] clusters = new Clusters[nFrags];
			SumPoints[] partialResult = new SumPoints[nFrags];
			int curFragment = 0;
			for (String fragmentPath : fragmentsPaths) {
				if (doDebug) {
					init = System.nanoTime();
				}
				if (!passPaths) {
					Fragment fragment = dataSet.get(fragmentPath);
					// clusters[curFragment] = clusters_points_partial(fragment,
					// muResult, K,
					// vectorsPerFragment * curFragment);
					// partialResult[curFragment] = partial_sum(fragment,
					// clusters[curFragment], K, vectorsPerFragment *
					// curFragment);
					partialResult[curFragment] = clusters_points_and_partial_sum(fragment, muResult, K,
							vectorsPerFragment * curFragment);
				} else {
					partialResult[curFragment] = clusters_points_and_partial_sum2(fragmentPath, muResult, K,
							vectorsPerFragment * curFragment);
				}
				if (doDebug) {
					end = System.nanoTime();
					System.out.println("[TIMER] MAP TASK : " + (end - init) / 1000 + " micros");
					System.out.println("[LOG] mu id after map " + muResult.getID());
				}
				curFragment++;
			}

			// MERGE-REDUCE
			LinkedList<Integer> q = new LinkedList<Integer>();
			for (int i = 0; i < nFrags; i++) {
				q.add(i);
			}

			int x = 0;
			while (!q.isEmpty()) {
				x = q.poll();
				int y;
				if (!q.isEmpty()) {
					y = q.poll();
					// with RETURN
					if (doDebug) {
						init = System.nanoTime();
					}
					partialResult[x] = reduceCentersTask(partialResult[x], partialResult[y]);
					if (doDebug) {
						end = System.nanoTime();
						System.out.println("[TIMER] REDUCE TASK : " + (end - init) / 1000 + " micros");
					}
					// with INOUT
					// reduceCentersTask(partialResult[x], partialResult[y]);
					q.add(x);
				}
			}

			// NORMALIZE clusters
			muResult = new Fragment(partialResult[0].normalize());

			++currentIteration;
		}

		iterationsDone = currentIteration - 1;
	}

	public static SumPoints clusters_points_and_partial_sum2(String fragmentPath, Fragment mu, int k, int ind) {
		Fragment fragment = new Fragment();
		fragment.readFromFile(fragmentPath, vectorsPerFragment, dimsPerVector);

		// clusters_points_partial
		Clusters clustersOfFrag = new Clusters(k);

		int numDimensions = fragment.getDimensionsPerVector();
		for (int p = 0; p < fragment.getNumVectors(); p++) {
			int closest = -1;
			float closestDist = Float.MAX_VALUE;
			for (int m = 0; m < mu.getNumVectors(); m++) {
				float dist = 0;
				for (int dim = 0; dim < numDimensions; dim++) {
					float tmp = fragment.getPoint(p, dim) - mu.getPoint(m, dim);
					dist += tmp * tmp;
				}
				if (dist < closestDist) {
					closestDist = dist;
					closest = m; // cluster it belongs to
				}
			}
			int value = ind + p;
			clustersOfFrag.addIndex(closest, value);
		}

		// partial_sum
		SumPoints pSum = new SumPoints(k, fragment.getDimensionsPerVector());
		for (int c = 0; c < clustersOfFrag.getSize(); c++) {
			int[] positionValues = clustersOfFrag.getIndexes(c);
			for (int i = 0; i < clustersOfFrag.getIndexesSize(c); i++) {
				int value = positionValues[i];
				float[] v = fragment.getVector(value - ind);
				pSum.sumValue(c, v, 1);
			}
		}
		return pSum;
	}

	public static SumPoints clusters_points_and_partial_sum(Fragment fragment, Fragment mu, int k, int ind) {

		// clusters_points_partial
		Clusters clustersOfFrag = new Clusters(k);

		int numDimensions = fragment.getDimensionsPerVector();
		for (int p = 0; p < fragment.getNumVectors(); p++) {
			int closest = -1;
			float closestDist = Float.MAX_VALUE;
			for (int m = 0; m < mu.getNumVectors(); m++) {
				float dist = 0;
				for (int dim = 0; dim < numDimensions; dim++) {
					float tmp = fragment.getPoint(p, dim) - mu.getPoint(m, dim);
					dist += tmp * tmp;
				}
				if (dist < closestDist) {
					closestDist = dist;
					closest = m; // cluster it belongs to
				}
			}
			int value = ind + p;
			clustersOfFrag.addIndex(closest, value);
		}

		// partial_sum
		SumPoints pSum = new SumPoints(k, fragment.getDimensionsPerVector());
		for (int c = 0; c < clustersOfFrag.getSize(); c++) {
			int[] positionValues = clustersOfFrag.getIndexes(c);
			for (int i = 0; i < clustersOfFrag.getIndexesSize(c); i++) {
				int value = positionValues[i];
				float[] v = fragment.getVector(value - ind);
				pSum.sumValue(c, v, 1);
			}
		}
		return pSum;
	}

	/**
	 * Reduce task with IN parameters returning the result
	 */
	public static SumPoints reduceCentersTask(SumPoints a, SumPoints b) {
		for (int i = 0; i < b.getSize(); i++) {
			a.sumValue(i, b.getValue(i), b.getNumPoints(i));
		}
		return a;
	}

	/**
	 * Checks arguments for the application
	 * 
	 * @param args
	 *            user arguments
	 */
	private static void checkArguments(String[] args) {
		if (args.length < 1) {
			System.err.println("[ERROR] Bad arguments");
			getUsage();
			System.exit(1);
		}
		if (args[0].equals("-h")) {
			getUsage();
			System.exit(0);
		} else {
			fragmentsDirPath = args[0];
		}

		for (int argIndex = 1; argIndex < args.length;) {
			String arg = args[argIndex++];
			if (arg.equals("-k")) {
				K = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-iterations")) {
				iterations = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-sizefrag")) {
				vectorsPerFragment = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-dimensions")) {
				dimsPerVector = Integer.parseInt(args[argIndex++]);
			} else if (arg.equals("-preread")) {
				preRead = true;
			} else if (arg.equals("-passpaths")) {
				passPaths = true;
			} else if (arg.equals("-debug")) {
				doDebug = true;
			} else if (arg.equals("-h")) {
				getUsage();
				System.exit(0);
			}
		}
		if (passPaths && preRead) {
			System.err.println("[ERROR] -passpaths and -preread flags cannot be used together");
			getUsage();
			System.exit(1);
		}
		if (passPaths) {
			preRead = false;
		}
	}

	/**
	 * Retrieves usage information
	 */
	private static void getUsage() {
		System.out.println("\n\n    Usage: " + KMeansFiles.class.getName() + " fragments_dir_path "
				+ "[-k clusters ] [-iterations maxIterations ] [-sizefrag vectorsPerFragment ] "
				+ "[-dimensions dimsPerVector ] [-preread] [-passpaths (=> !preRead)] [-debug] \n\n");
	}

	// @task
	/*
	 * public static Clusters clusters_points_partial(Fragment points, Fragment
	 * mu, int k, int ind) { Clusters clustersOfFrag = new Clusters(k);
	 * 
	 * int numDimensions = points.getDimensionsPerVector(); for (int p = 0; p <
	 * points.getNumVectors(); p++) { int closest = -1; float closestDist =
	 * Float.MAX_VALUE; for (int m = 0; m < mu.getNumVectors(); m++) { float
	 * dist = 0; for (int dim = 0; dim < numDimensions; dim++) { float tmp =
	 * points.getPoint(p, dim) - mu.getPoint(m, dim); dist += tmp * tmp; } if
	 * (dist < closestDist) { closestDist = dist; closest = m; // cluster it
	 * belongs to } } int value = ind + p; clustersOfFrag.addIndex(closest,
	 * value); }
	 * 
	 * return clustersOfFrag; }
	 */

	// @task
	/*
	 * public static SumPoints partial_sum(Fragment points, Clusters cluster,
	 * int k, int ind) { SumPoints pSum = new SumPoints(k,
	 * points.getDimensionsPerVector()); for (int c = 0; c < cluster.getSize();
	 * c++) { int[] positionValues = cluster.getIndexes(c); for (int i = 0; i <
	 * cluster.getIndexesSize(c); i++) { int value = positionValues[i]; float[]
	 * v = points.getVector(value - ind); pSum.sumValue(c, v, 1); } } return
	 * pSum; }
	 */

	// @task
	/*
	 * // with INOUT public static void reduceCentersTask(SumPoints a, SumPoints
	 * b){ for (int i = 0; i < b.getSize(); i++){ a.sumValue(i, b.getValue(i),
	 * b.getNumPoints(i)); } }
	 */

}
