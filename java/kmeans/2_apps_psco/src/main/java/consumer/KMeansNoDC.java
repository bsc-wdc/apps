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

import java.util.Arrays;
// import java.util.LinkedList;

import model.Clusters;
import model.Fragment;
import model.SumPoints;
import randomness.Randomness;

// UNCOMMENT THIS IF YOU UNCOMMENT COMPSs.waitForAllTasks(); statements
 import es.bsc.compss.api.COMPSs;

public class KMeansNoDC {
	static int K = 4; // num of clusters
	static double epsilon = 1e-4; // convergence criteria
	static int iterations = 50; // max iterations to converge

	static int randomFragments = 0; // number of random fragments
	static int vectorsPerFragment = 2; // max vectors per fragment
	static int dimsPerVector = 2; // max dimensions per vector

	static boolean preCreate = false;
	static boolean doDebug = false;

	// static Fragment muResult; // final result

	public static void main(String[] args) {
		System.out.println("[LOG] Executing KMeans with fragments in files");

		checkArguments(args);

		System.out.println("[LOG] Running with the following parameters:");
		System.out.println("- Clusters      : " + K);
		System.out.println("- Iterations    : " + iterations);
		System.out.println("- Random frags  : " + randomFragments);
		System.out.println("- VectorsPerFrag: " + vectorsPerFragment);
		System.out.println("- Dimensions    : " + dimsPerVector);
		System.out.println("- Pre-create    : " + preCreate);

		// Mu generation
		// Fragment muResult = new Fragment(K, dimsPerVector); // random
		// muResult.fillPoints(Randomness.muSeed);
		// System.out.println("[LOG] Random mu, 1st vector:" +
		// Arrays.toString(muResult.getVector(0)));

		Fragment[] fragments = new Fragment[randomFragments];

		// Precreate fragments if required
		if (preCreate) {
			for (int i = 0; i < randomFragments; i++) {
				Fragment f = createFragment(vectorsPerFragment, dimsPerVector, Randomness.nextInt());
				fragments[i] = f;
			}
			System.out.println("[LOG] Pre-Created " + randomFragments + " random fragments");
		}

		// KMeans execution
		System.out.println("[LOG] Init computeKMeans");
		long startTime = System.currentTimeMillis();
		// computeKMeans(fragments, muResult);
		computeKMeans(fragments);
		 COMPSs.waitForAllTasks();
		long estimatedTime = System.currentTimeMillis() - startTime;
		System.out.println("[TIMER] Elapsed time: " + estimatedTime);
	}

	public static Fragment createFragment(int vectors, int dimensions, int seed) {
		Fragment f = new Fragment(vectors, dimensions);
		f.fillPoints(seed);
		if (doDebug) {
			System.out.println("[LOG] Generated random fragment with seed " + seed + ". 1st vector "
					+ Arrays.toString(f.getVector(0)));
		}
		return f;
	}

	private static void computeKMeans(Fragment[] preCreatedfragments) {
		long init = 0, end;
		Fragment muResult = new Fragment(K, dimsPerVector);
		muResult.fillPoints(Randomness.muSeed);
		System.out.println("[LOG] Random mu");
		printMuResult(muResult);

		Fragment[] fragments;
		if (!preCreate) {
			// Create fragments if not precreated
			if (doDebug) {
				init = System.nanoTime();
			}
			fragments = new Fragment[randomFragments];
			for (int i = 0; i < randomFragments; i++) {
				Fragment f = createFragment(vectorsPerFragment, dimsPerVector, Randomness.nextInt());
				fragments[i] = f;
				if (doDebug) {
					printMuResult(fragments[i]);
				}
			}
			if (doDebug) {
				end = System.nanoTime();
				System.out.println("[LOG] Created " + randomFragments + " in " + (end - init) / 1000 + " micros");
			}
		} else {
			fragments = preCreatedfragments;
		}

		Fragment oldmu = null;
		int currentIteration = 1;
		// Convergence condition
		while (currentIteration <= iterations && !Utils.has_converged(muResult, oldmu, epsilon)) {
			System.out.println("[LOG] Starting iteration " + currentIteration + " of " + iterations);

			oldmu = muResult;
			// Clusters[] clusters = new Clusters[randomFragments];
			SumPoints[] partialResult = new SumPoints[randomFragments];
			int curFragment = 0;
			for (Fragment fragment : fragments) {
				if (doDebug) {
					init = System.nanoTime();
				}
				// clusters[curFragment] = clusters_points_partial(fragment,
				// muResult, K,
				// vectorsPerFragment * curFragment);
				// partialResult[curFragment] = partial_sum(fragment,
				// clusters[curFragment], K,
				// vectorsPerFragment * curFragment);
				if (doDebug) {
					printMuResult(fragment);
				}
				partialResult[curFragment] = clusters_points_and_partial_sum(fragment, muResult, K,
						vectorsPerFragment * curFragment);

				if (doDebug) {
					end = System.nanoTime();
					System.out.println("[TIMER] MAP TASK : " + (end - init) / 1000 + " micros");
				}
				curFragment++;
			}

			// MERGE-REDUCE
			int neighbor = 1;
			while (neighbor < randomFragments) {
				for (int i = 0; i < randomFragments; i += 2 * neighbor) {
					if (i + neighbor < randomFragments) {
						partialResult[i] = reduceCentersTask(partialResult[i], partialResult[i + neighbor]);
					}
				}
				neighbor *= 2;
			}

			// NORMALIZE clusters
			if (doDebug) {
				System.out.println("[LOG] Normalizing result of iteration " + currentIteration);
			}
			muResult = new Fragment(partialResult[0].normalize());

			++currentIteration;
		}

		System.out.println("[RESULT] Iterated " + (currentIteration - 1) + " times to get mu:");
		printMuResult(muResult);
	}

	private static void printMuResult(Fragment muResult) {
		System.out.println("[MU]:");
		for (int i = 0; i < K; i++) {
			System.out.println("   - " + Arrays.toString(muResult.getVector(i)));
		}
	}

	public static SumPoints clusters_points_and_partial_sum(Fragment fragment, Fragment mu, int k, int ind) {

		// clusters_points_partial
		Clusters clustersOfFrag = new Clusters(k);

		int numDimensions = fragment.getDimensionsPerVector();
		int numVectors = fragment.getNumVectors();
		for (int p = 0; p < numVectors; p++) {
			int closest = -1;
			float closestDist = Float.MAX_VALUE;
			for (int m = 0; m < k; m++) {
				float dist = 0;
				for (int dim = 0; dim < numDimensions; dim++) {
					float aux = fragment.getPoint(p, dim);
					float tmp = aux - mu.getPoint(m, dim);
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
		// SumPoints pSum = new SumPoints(k, fragment.getDimensionsPerVector());
		SumPoints pSum = new SumPoints(k, numDimensions);
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
			randomFragments = Integer.parseInt(args[0]);
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
			} else if (arg.equals("-precreate")) {
				preCreate = true;
			} else if (arg.equals("-debug")) {
				doDebug = true;
			} else if (arg.equals("-h")) {
				getUsage();
				System.exit(0);
			}
		}

	}

	private static void getUsage() {
		System.out.println("\n\n    Usage: " + KMeansNoDC.class.getName() + " nRandomFragments "
				+ "[-k clusters] [-iterations maxIterations] [-sizefrag vectorsPerFragment] "
				+ "[-dimensions dimsPerVector] [-precreate] [-debug] \n\n");
	}

	// public static Clusters clusters_points_partial(Fragment points, Fragment
	// mu, int k, int ind) {
	// Clusters clustersOfFrag = new Clusters(k);
	// int numDimensions = points.getDimensionsPerVector();
	// for (int p = 0; p < points.getNumVectors(); p++) {
	// int closest = -1;
	// float closestDist = Float.MAX_VALUE;
	// for (int m = 0; m < mu.getNumVectors(); m++) {
	// float dist = 0;
	// for (int dim = 0; dim < numDimensions; dim++) {
	// float tmp = points.getPoint(p, dim) - mu.getPoint(m, dim);
	// dist += tmp * tmp;
	// }
	// if (dist < closestDist) {
	// closestDist = dist;
	// closest = m; // cluster it belongs to
	// }
	// }
	// int value = ind + p;
	// clustersOfFrag.addIndex(closest, value);
	// }
	// return clustersOfFrag;
	// }

	// public static SumPoints partial_sum(Fragment points, Clusters cluster,
	// int k, int ind) {
	// SumPoints pSum = new SumPoints(k, points.getDimensionsPerVector());
	// for (int c = 0; c < cluster.getSize(); c++) {
	// int[] positionValues = cluster.getIndexes(c);
	// for (int i = 0; i < cluster.getIndexesSize(c); i++) {
	// int value = positionValues[i];
	// float[] v = points.getVector(value - ind);
	// pSum.sumValue(c, v, 1);
	// }
	// }
	// return pSum;
	// }

	// @task
	// with INOUT
	// public static void reduceCentersTask(SumPoints a, SumPoints b) {
	// for (int i = 0; i < b.getSize(); i++) {
	// a.sumValue(i, b.getValue(i), b.getNumPoints(i));
	// }
	// }

}
