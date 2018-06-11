package producer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;

import model.Fragment;
import model.FragmentCollection;
import randomness.Randomness;
import storage.StorageItf;
import util.ids.StorageLocationID;

public class FragmentDataClayGenerator {
	public static void main(String[] args) throws Exception {
		if (args.length != 5) {
			System.err.println("[ERROR] Bad arguments. Usage: \n\n" + FragmentDataClayGenerator.class.getName()
					+ " <config_properties> <frag_col_name> <num_fragments> <vectors_per_fragment> <dimensions_per_vector> \n");
			return;
		}

		StorageItf.init(args[0]);

		String fragColAlias = args[1];
		int nFrags = Integer.parseInt(args[2]);
		int nVectorsPerFragment = Integer.parseInt(args[3]);
		int nDimensions = Integer.parseInt(args[4]);

		Set<StorageLocationID> locations = StorageItf.getBackends().keySet();
		ArrayList<StorageLocationID> sortedLocations = new ArrayList<StorageLocationID>(locations);
		int curLocation = 0;

		FragmentCollection fragmentCollection = new FragmentCollection();
		int seed = Randomness.nextInt();
		for (int i = 0; i < nFrags; i++) {
			Fragment f = new Fragment(nVectorsPerFragment, nDimensions);
			f.makePersistent(true, sortedLocations.get(curLocation));
			f.fillPoints(seed);
			fragmentCollection.addFragment(f);
			curLocation++;
			if (curLocation == sortedLocations.size()) {
				curLocation = 0;
			}
			System.out.println("[LOG] Created fragment " + f.getID() + " with seed " + seed + " at " + f.getLocation()
					+ " with " + f.getNumVectors() + " vectors." + ". First vector " + Arrays.toString(f.getVector(0)));
			seed = Randomness.nextInt();
		}
		fragmentCollection.makePersistent(fragColAlias);
		System.out.println("[LOG] FragmentCollection " + fragmentCollection.getID() + " created at "
				+ fragmentCollection.getLocation() + " with " + fragmentCollection.getNumFragments() + " fragments");

		StorageItf.finish();
	}

}
