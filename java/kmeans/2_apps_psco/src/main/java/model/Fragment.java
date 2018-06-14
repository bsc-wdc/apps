package model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Serializable;
import java.util.Random;

import serialization.DataClayObject;


/**
 * A class to encapsulate an input set of Points for use in the KMeans program
 * and the reading/writing of a set of points to data files.
 */
@SuppressWarnings("serial")
public class Fragment extends DataClayObject implements Serializable {
    
	private float[][] points;

	// FOR COMPSS
	public Fragment() {

	}
	
	// FOR COMPSS
	public float[][] getPoints() {
		return points;
	}

	// FOR COMPSS
	public void setPoints(float[][] pts) {
		points = new float[pts.length][];
		for (int i = 0; i < pts.length; i++) {
			points[i] = new float[pts[i].length];
			for (int j = 0; j < pts[i].length; j++) {
				points[i][j] = pts[i][j];
			}
		}
		
		// TODO points = pts; // for COMPSs
	}


	public Fragment(float[][] pts) {
		setPoints(pts);
	}

	public Fragment(int foo, final String alias) {
		super(alias);
	}

	/**
	 * Constructor for random generation
	 * 
	 * @param idForSeed
	 *            seed for random generator
	 * @param sizeFrag
	 *            vectors for the fragment
	 * @param dimsPerVector
	 *            dimensions per vector
	 */
	public Fragment(int sizeFrag, int dimsPerVector) {
		this.points = new float[sizeFrag][dimsPerVector];
	}

	/**
	 * Update points with given points
	 * 
	 * @param pts
	 */
	public void updatePoints(float[][] pts) {
		points = new float[pts.length][];
		for (int i = 0; i < pts.length; i++) {
			points[i] = new float[pts[i].length];
			for (int j = 0; j < pts[i].length; j++) {
				points[i][j] = pts[i][j];
			}
		}
	}

	public void fillPoints(int idForSeed) {
		Random random = new Random(idForSeed);
		for (int i = 0; i < points.length; i++) {
			for (int j = 0; j < points[i].length; j++) {
				// Random between [-1,1)
				points[i][j] = random.nextFloat() * 2 - 1;
			}
		}
	}

	public float getPoint(int v, int dim) {
		return this.points[v][dim];
	}

	public float[] getVector(int v) {
		return this.points[v];
	}

	public int getNumVectors() {
		return this.points.length;
	}

	public int getDimensionsPerVector() {
		return this.points[0].length;
	}

	public void dumpToFile(String filePath) throws IOException {
		FileWriter fw = null;
		BufferedWriter bw = null;

		fw = new FileWriter(filePath);
		bw = new BufferedWriter(fw);
		String strPoint = new String();
		for (int i = 0; i < points.length; i++) {
			for (int j = 0; j < points[i].length; j++) {
				strPoint = strPoint + points[i][j] + " ";
			}
			bw.write(strPoint);
			bw.newLine();
			strPoint = "";
		}
		bw.close();
		fw.close();
	}

	public void readFromFile(String filepath, int vectorsToGet, int dimsPerVector) {
		points = new float[vectorsToGet][dimsPerVector];

		FileReader fr = null;
		BufferedReader br = null;
		try {
			fr = new FileReader(filepath);
			br = new BufferedReader(fr);

			for (int i = 0; i < vectorsToGet; i++) {
				String line = br.readLine();
				String[] values = line.split(" ");
				for (int j = 0; j < dimsPerVector; j++) {
					points[i][j] = Float.valueOf(values[j]);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (fr != null) {
				try {
					fr.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
			if (br != null) {
				try {
					br.close();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	/**
	 * Merged cluster_points_partial and partial_sum.
	 * 
	 * @param mu
	 *            current mu
	 * @param k
	 *            number of total clusters
	 * @param ind
	 *            current index
	 * @return the resulting SumPoints
	 */
	public SumPoints cluster_points_and_partial_sum(Fragment mu, int k, int ind) {
		Clusters cluster = new Clusters(k);

		int numDimensions = this.getDimensionsPerVector();
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
			cluster.addIndex(closest, value);
		}

		SumPoints pSum = new SumPoints(k, numDimensions);
		// cluster.getSize = k
		for (int c = 0; c < cluster.getSize(); c++) {
			int[] positionValues = cluster.getIndexes(c);
			for (int i = 0; i < cluster.getIndexesSize(c); i++) {
				int value = positionValues[i];
				float[] v = this.getVector(value - ind);
				pSum.sumValue(c, v, 1);
			}
		}

		// StorageLocationID location = this.getLocation();
		pSum.makePersistent(true, null);

		return pSum;
	}

	// public Clusters clusters_points_partial(Fragment mu, int k, int ind) {
	// Clusters clustersOfFrag = new Clusters(k);
	//
	// int numDimensions = this.getDimensionsPerVector();
	// for (int p = 0; p < this.getNumVectors(); p++) {
	// int closest = -1;
	// float closestDist = Float.MAX_VALUE;
	// for (int m = 0; m < mu.getNumVectors(); m++) {
	// float dist = 0;
	// for (int dim = 0; dim < numDimensions; dim++) {
	// float tmp = this.getPoint(p, dim) - mu.getPoint(m, dim);
	// dist += tmp * tmp;
	// }
	// if (dist < closestDist) {
	// closestDist = dist;
	// closest = m; // cluster al que pertenece
	// }
	// }
	// int value = ind + p;
	// clustersOfFrag.addIndex(closest, value);
	// }
	// StorageLocationID location = this.getLocation();
	// clustersOfFrag.makePersistent(true, location);
	//
	// return clustersOfFrag;
	// }

	// public SumPoints partial_sum(Clusters cluster, int k, int ind) {
	// SumPoints pSum = new SumPoints(k, this.getDimensionsPerVector());
	// // cluster.getSize = k
	// for (int c = 0; c < cluster.getSize(); c++) {
	// int[] positionValues = cluster.getIndexes(c);
	// for (int i = 0; i < cluster.getIndexesSize(c); i++) {
	// int value = positionValues[i];
	// float[] v = this.getVector(value - ind);
	// pSum.sumValue(c, v, 1);
	// }
	// }
	// pSum.makePersistent(true, null);
	// return pSum;
	// }

}
