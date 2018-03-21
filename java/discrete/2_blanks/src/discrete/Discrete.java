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
package discrete;

import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedList;
import java.util.Queue;
import java.util.UUID;

public class Discrete {

	private static final String PYDOCK = "/1B6C.pydock";
	private static final String LECTURA = "/lectura.dat";
	private static final String DMD_PARAM = "/dmdparm.dat";

	private static final String TSNAP = "TSNAP";
	private static final String NATOM = "NATOM1";
	private static final String NBLOC = "NBLOC";
	private static final String TEMP = "TEMP";
	private static final String SEED = "SEED";

	private static final int N = 5;
	private static final float FVDW_MAX = 2;
	private static final float FSOLV_MAX = 7;
	private static final float EPS_MAX = 5;
	private static final int FVDW_STEPS = 1;
	private static final int FSOLV_STEPS = 1;
	private static final int EPS_STEPS = 2;
	private static final float FVDW_STEP = FVDW_MAX / FVDW_STEPS;
	private static final float FSOLV_STEP = FSOLV_MAX / FSOLV_STEPS;
	private static final float EPS_STEP = EPS_MAX / EPS_STEPS;

	private static String natom = null;
	private static String partialParams;

	private static boolean debug;
	private static String binDir;
	private static String dataDir;
	private static String structDir;
	private static String tmpDir;

	public static void main(String args[]) throws Exception {

		debug = Boolean.parseBoolean(args[0]);
		binDir = args[1];
		dataDir = args[2];
		structDir = args[3];
		tmpDir = args[4];
		String scoreDir = args[5];

		if (debug) {
			System.out.println("Parameters: ");
			System.out.println("- Debug Enabled");
			System.out.println("- Binary directory: " + binDir);
			System.out.println("- Data directory: " + dataDir);
			System.out.println("- Structures directory: " + structDir);
			System.out.println("- Temporary directory: " + tmpDir);
			System.out.println("- Scores directory: " + scoreDir);
			System.out.println();
		}

		// read some parameters from files
		initParams();

		// BEGIN SETUP SECTION

		// generate receptor and ligand and run setup
		for (int i = 1; i <= N; i++) {
			String structure = structDir + "/1B6C_" + i + ".pdb";
			String receptor = tmpDir + "/receptor_" + i;
			// TODO Create file name for ligand
			String coordinates = tmpDir + "/coordinates_" + i;
			String topology = tmpDir + "/topology_" + i;

			// TODO Generate receptor & ligand
			// TODO Run DMD Setup
		}

		// END SETUP SECTION

		// BEGIN PARAMETER SWEEP SECTION

		Queue<String> coeffList = new LinkedList<String>();
		String pydock = dataDir + PYDOCK;
		Queue<String> list = new LinkedList<String>();

		// parameter sweep loops
		for (int i = 1; i <= FVDW_STEPS; i++) {

			double fvdw = i * FVDW_STEP;

			for (int j = 1; j <= FSOLV_STEPS; j++) {

				double fsolv = j * FSOLV_STEP;

				for (int k = 1; k <= EPS_STEPS; k++) {

					double eps = k * EPS_STEP;

					String params = genParamFile(fvdw, fsolv, eps);

					// TODO run N simulations for each configuration
					// Hint: variables params, natom, pydock and scoreDir are
					// declared and initialized

					// TODO Insert simulation results (average files) into
					// variable 'list'

					// merge all averages in a single file
					while (list.size() > 1) {
						Queue<String> listAux = new LinkedList<String>();

						while (list.size() > 1) {
							String av1 = list.poll();
							String av2 = list.poll();

							// TODO Run merge

							listAux.add(av1);
						}
						if (list.size() == 1)
							listAux.add(list.peek());

						list = listAux;
					}

					String score = scoreDir + "/score_" + fvdw + "_" + fsolv + "_" + eps + ".score";
					String coefficient = tmpDir + "/coeff_" + UUID.randomUUID();

					coeffList.add(coefficient);

					// generate the score file and calc the final coefficient of
					// the configuration
					DiscreteImpl.evaluate(list.poll(), pydock, fvdw, fsolv, eps, score, coefficient);
				}
			}
		}
		// END PARAMETER SWEEP SECTION

		// BEGIN FINAL SECTION

		// find out the min coefficient of all configurations
		while (coeffList.size() > 1) {
			String c1 = coeffList.poll();
			String c2 = coeffList.poll();

			DiscreteImpl.min(c1, c2);

			coeffList.add(c1);
		}

		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(coeffList.peek())));

		br.readLine();
		System.out.println("Best configuration:");
		System.out.println("FVDW FSOLV EPS");
		System.out.println("-----------------------------");
		System.out.println(br.readLine());

		br.close();

		// END FINAL SECTION
	}

	private static void initParams() throws IOException, FileNotFoundException, Exception {

		// read lectura file to extract TSNAP and NATOM
		String lecFile = dataDir + LECTURA;
		BufferedReader br;
		br = new BufferedReader(new FileReader(new File(lecFile)));

		String read;

		while ((read = br.readLine()) != null) {
			if (read.contains(NATOM))
				natom = read.substring(read.indexOf("=") + 1);
		}
		br.close();

		// read dmd_parm file to extract FVDW, FSOLV and EPS
		String parmFile = dataDir + DMD_PARAM;
		br = new BufferedReader(new FileReader(new File(parmFile)));

		String tsnap = null;
		String nbloc = null;
		String temp = null;
		String seed = null;

		while ((read = br.readLine()) != null) {
			if (read.contains(TSNAP))
				tsnap = read.substring(read.indexOf("=") + 1);
			else if (read.contains(NBLOC))
				nbloc = read.substring(read.indexOf("=") + 1);
			else if (read.contains(TEMP))
				temp = read.substring(read.indexOf("=") + 1);
			else if (read.contains(SEED))
				seed = read.substring(read.indexOf("=") + 1);
		}
		br.close();

		if (natom == null || tsnap == null || nbloc == null || temp == null || seed == null)
			throw new Exception("Error reading parameters:\n TSNAP= " + tsnap + "\n NBLOC= " + nbloc + "\n TEMP= "
					+ temp + "\n SEED= " + seed + "\n NATOM= " + natom);

		if (debug)
			System.out.println("Simulation Params:\n-TSNAP= " + tsnap + "\n-NBLOC= " + nbloc + "\n-TEMP= " + temp
					+ "\n-SEED= " + seed + "\n-NATOM1= " + natom);

		partialParams = " &INPUT\n  TSNAP=" + tsnap + "\n  NBLOC=" + nbloc + "\n  TEMP=" + temp + "\n  SEED="
				+ seed;
	}

	private static String genParamFile(double fvdw, double fsolv, double eps) throws IOException {
		String content = partialParams + "\n  FVDW=" + fvdw + "\n  FSOLV=" + fsolv + "\n  EPS=" + eps
				+ "\n &END\n";

		String path = tmpDir + "/dmdparm_" + UUID.randomUUID();

		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(new File(path)));

		bos.write(content.getBytes());
		bos.close();

		return path;
	}
}
