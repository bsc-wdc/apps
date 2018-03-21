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

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.RandomAccessFile;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.UUID;

public class DiscreteImpl {

    private static final String PERL_FILE = "/PDBtoDISCRETE.pl";
    private static final String DMD_BIN = "/DMDSetup";
    private static final String PARAM_SETUP = "/parmsetup.dat";
    private static final String RES_LIB = "/dmd_reslibrary.dat";
    private static final String POT = "/dmd_potentials.dat";
    private static final String DISCRETE = "/discrete";
    private static final String READSNAP = "/readsnap";

    private static final String ATOM = "/atomtypes.dat";
    private static final String SNAP_POT = "/potentials.dat";
    private static final String PROMIG = "/promig3col";
    private static final String PATRO_TMP = "patro.pdb";

    private static final String I_ARG = " -i ";
    private static final String PERL_REC_ARG = " -r ";
    private static final String PERL_LIG_ARG = " -l ";
    private static final String SET_REC_ARG = " -pdbin ";
    private static final String SET_LIG_ARG = " -ligpdbin ";
    private static final String SET_RES_ARG = " -rlib ";
    private static final String SET_POT_ARG = " -pot ";
    private static final String SET_TOP_ARG = " -top ";
    private static final String SET_CRD_ARG = " -r ";
    private static final String DIS_TOP_ARG = " -top ";
    private static final String DIS_CRD_ARG = " -r ";
    private static final String DIS_EN_ARG = " -ener ";
    private static final String DIS_RST_ARG = " -rst ";
    private static final String DIS_TRAJ_ARG = " -traj ";
    private static final String DIS_IN_ARG = " -in ";

    public static void genReceptorLigand(String pdbFile, String binDir, String recFile,
	    String ligFile) throws IOException, InterruptedException, Exception {
	/*
	 * Generate receptor and ligand with perl
	 */

	String perlFile = binDir + PERL_FILE;

	String cmd = "perl " + perlFile + I_ARG + pdbFile + PERL_REC_ARG + recFile + PERL_LIG_ARG
		+ ligFile + " -rec A -lig B -rdir=.";

	System.out.println("Executing " + cmd);

	Process perlProc = Runtime.getRuntime().exec(cmd);
	int exit = perlProc.waitFor();

	if (exit != 0)
	    launchException(perlProc.getErrorStream(), perlFile, exit);
    }

    public static void dmdSetup(String recFile, String ligFile, String binDir, String dataDir,
	    String topFile, String crdFile) throws IOException, InterruptedException, Exception {
	/*
	 * Run DMDSetup
	 */

	String dmdBinary = binDir + DMD_BIN;
	String paramSetupFile = dataDir + PARAM_SETUP;
	String resLibFile = dataDir + RES_LIB;
	String potFile = dataDir + POT;

	String cmd = dmdBinary + I_ARG + paramSetupFile + SET_REC_ARG + recFile + SET_LIG_ARG
		+ ligFile + SET_RES_ARG + resLibFile + SET_POT_ARG + potFile + SET_TOP_ARG
		+ topFile + SET_CRD_ARG + crdFile;

	System.out.println("Executing " + cmd);

	Process dmdProcess = Runtime.getRuntime().exec(cmd);
	int exit = dmdProcess.waitFor();

	System.out.println("Exit value of dmdSetup: " + exit + " with outputs " + topFile + " "
		+ crdFile);

	if (exit != 0) {
	    launchException(dmdProcess.getErrorStream(), dmdBinary, exit);
	}
	System.out.println("Ending dmdSetup task with outputs: " + topFile + " " + crdFile);
    }

    public static void simulate(String paramFile, String topFile, String crdFile, String natom,
	    String binDir, String dataDir, String averageFile) throws IOException,
	    InterruptedException, Exception {

	// try {
	// delete the tmp files when done
	new File(PATRO_TMP).deleteOnExit();

	/*
	 * Run discrete
	 */

	String discreteBinary = binDir + DISCRETE;

	String enFile = "energy_" + UUID.randomUUID();
	String restartFile = "restart_" + UUID.randomUUID();
	String trajFile = "trajectory_" + UUID.randomUUID();
	String inputFile = "input_" + UUID.randomUUID();

	new File(enFile).deleteOnExit();
	new File(restartFile).deleteOnExit();
	new File(trajFile).deleteOnExit();
	new File(inputFile).deleteOnExit();

	String cmd = discreteBinary + I_ARG + paramFile + DIS_TOP_ARG + topFile + DIS_CRD_ARG
		+ crdFile + DIS_EN_ARG + enFile + DIS_RST_ARG + restartFile + DIS_TRAJ_ARG
		+ trajFile + DIS_IN_ARG + inputFile;

	System.out.println("Executing " + cmd);

	System.out.println("DISCRETE FILES: " + new File(paramFile).length() + " "
		+ new File(topFile).length() + " " + new File(crdFile).length());

	Process discreteProc = Runtime.getRuntime().exec(cmd);
	// System.out.println("WAITING");
	// int exit = discreteProc.waitFor();
	// System.out.println("EXECUTED");

	ProcessHandler ph = new ProcessHandler(discreteProc, null, System.err);
	int exit = ph.waitFor();

	if (exit != 0)
	    launchException(discreteProc.getErrorStream(), discreteBinary, exit);

	System.out.println("Done");

	/*
	 * Run readsnap
	 */

	// TODO
	System.out.println("FILES EXIST: " + new File(enFile).length() + " "
		+ new File(restartFile).length() + " " + new File(trajFile).length() + " "
		+ new File(inputFile).length());

	String atomFile = dataDir + ATOM;
	String snapPotFile = dataDir + SNAP_POT;

	// build the lectura file
	String lec = " &INPUT\n ,FILE9='" + inputFile + "'\n ,FILE20='" + trajFile
		+ "'\n ,FILE16='" + atomFile + "'\n ,FILE17='" + snapPotFile
		+ "'\n ,TSNAP=10000\n ,NATOM1=" + natom + "\n &END\n";

	String readsnapBinary = binDir + READSNAP;

	System.out.println("Executing " + readsnapBinary + " with:\n" + lec);

	Process snapProc = Runtime.getRuntime().exec(readsnapBinary);

	// pass the new lecFile as standard input
	BufferedOutputStream stdin;

	stdin = new BufferedOutputStream(snapProc.getOutputStream());
	stdin.write(lec.getBytes());
	stdin.flush();
	stdin.close();

	exit = snapProc.waitFor();

	if (exit != 0)
	    launchException(snapProc.getErrorStream(), readsnapBinary, exit);

	String enecontFile = "enecont_" + UUID.randomUUID();

	new File(enecontFile).deleteOnExit();

	BufferedInputStream stdout;
	BufferedOutputStream enecont;

	stdout = new BufferedInputStream(snapProc.getInputStream());
	enecont = new BufferedOutputStream(new FileOutputStream(enecontFile));

	int bytes;
	byte[] b = new byte[1024];

	// write the output to enecont file
	while ((bytes = stdout.read(b)) >= 0)
	    enecont.write(b, 0, bytes);

	stdout.close();
	enecont.close();

	/*
	 * Run promig3col
	 */

	// get the last lines of the enecont file
	Process tailProc;

	tailProc = Runtime.getRuntime().exec("tail -n50 " + enecontFile);
	tailProc.waitFor();

	// execute promig with the last lines as input
	String promigBinary = binDir + PROMIG;

	System.out.println("Executing " + promigBinary);

	Process promigProc = Runtime.getRuntime().exec(promigBinary);

	stdin = new BufferedOutputStream(promigProc.getOutputStream());
	stdout = new BufferedInputStream(tailProc.getInputStream());

	while ((bytes = stdout.read(b)) >= 0)
	    stdin.write(b, 0, bytes);

	stdout.close();
	stdin.close();

	exit = promigProc.waitFor();

	if (exit != 0)
	    launchException(promigProc.getErrorStream(), promigBinary, exit);

	// calculate score with promig output and generate the partial score
	// file
	BufferedReader br;

	br = new BufferedReader(new InputStreamReader(promigProc.getInputStream()));

	String promigLine = br.readLine() + "\n";

	FileOutputStream result = new FileOutputStream(averageFile);
	result.write(promigLine.getBytes());
	result.close();

	System.out.println("DONE");
	// } catch (Throwable e) {
	// e.printStackTrace();
	// System.out.println("ERROR!!!");
	// }
    }

    // merge the files f1 and f2 into f1
    public static void merge(String f1, String f2) throws FileNotFoundException, IOException {

	BufferedInputStream bis;
	RandomAccessFile file;

	bis = new BufferedInputStream(new FileInputStream(f2));
	file = new RandomAccessFile(new File(f1), "rw");
	file.seek(file.length());

	byte[] b = new byte[1024];
	int bytes;

	while ((bytes = bis.read(b)) >= 0)
	    file.write(b, 0, bytes);

	file.close();
	bis.close();
    }

    // calculates the coefficient and writes the score file of a set of
    // simulations
    public static void evaluate(String averageFile, String pydockFile, double fvdw, double fsolv,
	    double eps, String scoreFile, String coeffFile) throws IOException,
	    FileNotFoundException {

	// double dfvdw = Double.valueOf(conf.fvdw);
	// double dfsolv = Double.valueOf(conf.fsolv);
	// double deps = Double.valueOf(conf.eps);

	BufferedReader pydock, average;
	FileOutputStream score;

	pydock = new BufferedReader(new InputStreamReader(new FileInputStream(pydockFile)));
	average = new BufferedReader(new InputStreamReader(new FileInputStream(averageFile)));
	score = new FileOutputStream(scoreFile);

	String line;
	double total = 0;
	int i = 0;

	SortedMap<Double, Double> scoreMap = new TreeMap<Double, Double>();

	// Calculates a coefficient dividing the sum of the scores of the 10
	// lines with less RMSD in the pydockFile by the total average score

	while ((line = average.readLine()) != null) {

	    String[] pyTokens = pydock.readLine().split("\\s+");
	    String[] avgTokens = line.split("\\s+");

	    double rmsd = Double.valueOf(pyTokens[1]);
	    double finalScore = Double.valueOf(avgTokens[1]) * fvdw + Double.valueOf(avgTokens[2])
		    * fsolv + Double.valueOf(avgTokens[3]) * eps;

	    total += finalScore;

	    scoreMap.put(rmsd, finalScore);

	    String scoreLine;
	    scoreLine = pyTokens[0] + " " + rmsd + " " + finalScore + "\n";

	    score.write(scoreLine.getBytes());

	    i++;
	}

	score.close();
	pydock.close();
	average.close();

	double rmsdTotal = 0;
	int n = (int) Math.min(1, scoreMap.size() * 0.1d);

	for (int j = 0; j < n; j++) {
	    double first = scoreMap.firstKey();
	    rmsdTotal += scoreMap.remove(first);
	}

	double coefficient = rmsdTotal / (total / i);
	String content = String.valueOf(coefficient) + "\n" + fvdw + " " + fsolv + " " + eps;

	FileOutputStream coeff = new FileOutputStream(coeffFile);
	coeff.write(content.getBytes());
	coeff.close();
    }

    public static void min(String f1, String f2) throws FileNotFoundException, IOException {

	BufferedReader br1;
	BufferedReader br2;

	br1 = new BufferedReader(new InputStreamReader(new FileInputStream(f1)));
	br2 = new BufferedReader(new InputStreamReader(new FileInputStream(f2)));

	double c1 = Double.parseDouble(br1.readLine());
	double c2 = Double.parseDouble(br2.readLine());

	br1.close();

	if (c1 > c2) {
	    FileOutputStream fos = new FileOutputStream(f1);
	    String content = String.valueOf(c2) + "\n" + br2.readLine();
	    fos.write(content.getBytes());

	    fos.close();
	}
	br2.close();
    }

    private static void launchException(InputStream err, String bin, int exit) throws IOException,
	    Exception {

	BufferedReader stderr = new BufferedReader(new InputStreamReader(err));
	String line;

	while ((line = stderr.readLine()) != null)
	    System.out.println(line);

	stderr.close();

	throw new Exception(bin + " exit value was: " + exit);
    }

    // private static void executeCommand(String cmd) throws IOException,
    // InterruptedException {
    // Process proc = Runtime.getRuntime().exec(cmd);
    //
    // /* handling the streams so that dead lock situation never occurs. */
    // ProcessHandler inputStream = new ProcessHandler(proc.getInputStream(),
    // "out");
    // ProcessHandler errorStream = new ProcessHandler(proc.getErrorStream(),
    // "err");
    //
    // /* start the stream threads */
    // inputStream.start();
    // errorStream.start();
    //
    // // printStream(proc.getInputStream(), "out");
    // // printStream(proc.getErrorStream(), "err");
    //
    // proc.waitFor();
    // }
}
