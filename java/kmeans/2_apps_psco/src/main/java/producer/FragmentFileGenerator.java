package producer;

import java.io.File;
import java.util.Arrays;

import model.Fragment;
import randomness.Randomness;

public class FragmentFileGenerator {
	public static final String FRAGMENTPREFIX = "fragment";

	public static void main(String[] args) throws Exception {
		if (args.length != 4) {
			System.err.println("[ERROR] Bad arguments. Usage: \n\n" + FragmentFileGenerator.class.getName()
					+ " <frag_dir_path> <num_fragments> <vectors_per_fragment> <dimensions_per_vector> \n");
			return;
		}

		String fragDirPath = args[0];
		int nFrags = Integer.parseInt(args[1]);
		int nVectorsPerFragment = Integer.parseInt(args[2]);
		int nDimensions = Integer.parseInt(args[3]);

		// Fragments are recreated to avoid remote interactions
		// Using the same seeds fragments must be the same
		int seed = Randomness.nextInt();
		if (fragDirPath != null) {
			for (int i = 1; i <= nFrags; i++) {
				String filepath = fragDirPath + File.separator + FRAGMENTPREFIX + i;
				Fragment f = new Fragment(nVectorsPerFragment, nDimensions);
				f.fillPoints(seed);
				f.dumpToFile(filepath);
				System.out.println("[LOG] Created fragment with seed " + seed + " at " + filepath + " with "
						+ f.getNumVectors() + " vectors." + " First vector " + Arrays.toString(f.getVector(0)));
				seed = Randomness.nextInt();
			}
		}
	}

}
