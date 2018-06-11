package consumer;

import java.util.Arrays;
import java.util.Comparator;

import model.Fragment;
import producer.FragmentFileGenerator;

public class Utils {

	public static boolean has_converged(Fragment mu, Fragment oldmu, double epsilon) {
		if (oldmu == null) {
			return false;
		}
		
		float aux = 0;
		// for each centroid
		for (int k = 0; k < mu.getNumVectors(); k++) {
			float dist = 0;
			// for each centroid dimension
			for (int dim = 0; dim < mu.getDimensionsPerVector(); dim++) {
				float tmp = oldmu.getPoint(k, dim) - mu.getPoint(k, dim);
				dist += tmp * tmp;
			}
			aux += dist;
		}
		if (aux < epsilon * epsilon) {
			System.out.println("[LOG] Distancia_T: " + aux);
			return true;
		} else {
			System.out.println("[LOG] Distancia_F: " + aux);
			return false;
		}
	}
	
	/**
	 * Sorts fragments paths by its numerical suffix
	 * 
	 * @param fragmentsPaths
	 */
	public static void sortFragmentPaths(String[] fragmentsPaths) {
		Arrays.sort(fragmentsPaths, new Comparator<String>() {
			public int compare(String f1, String f2) {
				try {
					int i1 = Integer.parseInt(f1.split(FragmentFileGenerator.FRAGMENTPREFIX)[1]);
					int i2 = Integer.parseInt(f2.split(FragmentFileGenerator.FRAGMENTPREFIX)[1]);
					return i1 - i2;
				} catch (NumberFormatException e) {
					throw new AssertionError(e);
				}
			}
		});
	}
}
