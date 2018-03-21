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
package worker.genedetection;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Map;
import java.util.UUID;
import java.util.zip.ZipEntry;
import java.util.zip.ZipException;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import util.ProcessHandler;

/**
 * 
 * @author amantsok
 */
public class GeneDetectionImpl {

	private static final String APP_PATH = "/home/user/workspace/genedetection";
	private static final String MAKE_BLAST = APP_PATH + "/binary/makeblastdb";
	private static final String BLAST = APP_PATH + "/binary/tblastn";
	private static final String BLAST_PARSE = APP_PATH
			+ "/binary/blastplus.parse.pl";
	private static final String BLAST2GENE = APP_PATH + "/binary/blast2gene.pl";
	private static final String FILTRA = APP_PATH
			+ "/binary/filtraBl2gIdCovScaff.pl";
	private static final String BL2G_PARSE = APP_PATH
			+ "/binary/parser_bl2g.pl";
	private static final String OVERLAP = APP_PATH
			+ "/binary/gb.range.overlap.pl";
	private static final String OVERLAP_PARSE = APP_PATH
			+ "/binary/parse.overlap_2.pl";
	private static final String GET_SUBSEQ = APP_PATH + "/binary/getsubseq.pl";
	private static final String GENEWISE = APP_PATH + "/binary/genewise";
	private static final String WISECFG = APP_PATH + "/wisecfg";

	// Blast parameters
	private static final String NUM_ALIGNMENTS = "15";
	private static final String XDROP_GAP_FINAL = "0";
	private static final String EVALUE = "10.0";
	private static final String GAP_EXTEND = "2";
	private static final String SEG = "no";
	private static final String MATRIX = "BLOSUM62";
	private static final String GAP_OPEN = "11";
	private static final String NUM_DESCR = "25";

	/*
	 * Takes a genome and generates a ZIP file containing a genome database.
	 * Uses makeblastdb binary.
	 */
	public static void runNCBIFormatdb(String genomeFile, String dbType,
			String inputType, String dbName, String dbPath, String zipFile)
			throws IOException {

		File zip = new File(zipFile);
		String currentDir = zip.getCanonicalFile().getParent();

		ArrayList<String> params = new ArrayList<String>();
		params.add(MAKE_BLAST);
		params.add("-in");
		params.add(genomeFile);
		params.add("-dbtype");
		params.add(dbType);
		params.add("-input_type");
		params.add(inputType);
		params.add("-out");
		params.add(currentDir + dbName);
		params.add("-title");
		params.add(dbPath);

		System.out.println("Executing command:");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");
		}
		System.out.println();

		ProcessBuilder pb = new ProcessBuilder(params);
		ProcessHandler ph = new ProcessHandler(pb.start(), System.out,
				System.err);
		int exit = ph.waitFor();

		System.out.println("Exit status: " + exit);

		// Generate the ZIP file

		File nhr = new File(currentDir + "/blastdb.nhr");
		File nin = new File(currentDir + "/blastdb.nin");
		File nsq = new File(currentDir + "/blastdb.nsq");

		ArrayList<String> databaseFiles = new ArrayList<String>();

		databaseFiles.add(nhr.getCanonicalPath());
		databaseFiles.add(nin.getCanonicalPath());
		databaseFiles.add(nsq.getCanonicalPath());

		generateZipFile(databaseFiles, currentDir, zip.getName());

		nhr.delete();
		nin.delete();
		nsq.delete();
	}

	/*
	 * Unzips a genome database and runs tblastn against it for a protein,
	 * generating a report.
	 */
	public static void runNCBIBlastAgainstDB(String proteinFile, String dbName,
			String zipFile, String reportFile) throws ZipException, IOException {

		// Extract contents of ZIP file	
		File zip = new File(zipFile);
		File tmpDir = new File(zip.getParent(), "zip" + UUID.randomUUID());
		tmpDir.mkdir();
		tmpDir.deleteOnExit();
		extractFolder(zipFile, tmpDir);

		// Execute blast
		ArrayList<String> params = new ArrayList<String>();
		params.add(BLAST);
		params.add("-query");
		params.add(proteinFile);
		params.add("-db");
		params.add(tmpDir.getPath() + "/" + dbName);
		params.add("-num_alignments");
		params.add(NUM_ALIGNMENTS);
		params.add("-xdrop_gap_final");
		params.add(XDROP_GAP_FINAL);
		params.add("-evalue");
		params.add(EVALUE);
		params.add("-gapextend");
		params.add(GAP_EXTEND);
		params.add("-seg");
		params.add(SEG);
		params.add("-matrix");
		params.add(MATRIX);
		params.add("-gapopen");
		params.add(GAP_OPEN);
		params.add("-num_descriptions");
		params.add(NUM_DESCR);
		params.add("-out");
		params.add(reportFile);

		System.out.println("Executing command:");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");
		}
		System.out.println();

		ProcessBuilder pb = new ProcessBuilder(params);
		ProcessHandler ph = new ProcessHandler(pb.start(), System.out,
				System.err);
		int exit = ph.waitFor();

		System.out.println("Exit status: " + exit);
	}

	/*
	 * Takes a blast report and prepares it for blast2gene.
	 */
	public static void runBL2Gparse(String blastReport, String blastParsed)
			throws IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add("perl");
		params.add(BLAST_PARSE);
		params.add(blastReport);
		params.add(blastParsed);

		System.out.println("Executing command:");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");

		}
		System.out.println();

		ProcessBuilder pb = new ProcessBuilder(params);
		ProcessHandler ph = new ProcessHandler(pb.start(), System.out,
				System.err);
		int exit = ph.waitFor();

		System.out.println("Exit status: " + exit);
	}

	/*
	 * Runs blast2gene.
	 */
	public static void runBlast2Gene(String blastParsed, String bl2gFile)
			throws FileNotFoundException, IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add("perl");
		params.add(BLAST2GENE);
		params.add(blastParsed);

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");

		}
		System.out.println("> " + bl2gFile);

		ProcessBuilder pb = new ProcessBuilder(params);
		FileOutputStream fos = new FileOutputStream(bl2gFile);
		ProcessHandler ph = new ProcessHandler(pb.start(), fos, System.err);
		int exit = ph.waitFor();
		fos.close();

		System.out.println("Exit status: " + exit);
	}

	/*
	 * Filters the blast2gene output and generates a filtered file
	 */
	public static void filterBl2G(String bl2gFile, String filteredFile)
			throws IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add("perl");
		params.add(FILTRA);
		params.add(bl2gFile);
		params.add("0.0");
		params.add("50");
		params.add("Y");

		System.out.println("Executing command: ");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");
		}
		System.out.println("> " + filteredFile);

		ProcessBuilder pb = new ProcessBuilder(params);
		FileOutputStream fos = new FileOutputStream(filteredFile);
		ProcessHandler ph = new ProcessHandler(pb.start(), fos, System.err);
		int exit = ph.waitFor();
		fos.close();

		System.out.println("Exit status: " + exit);
	}

	/*
	 * Appends the contents of file2 to file1
	 */
	public static void mergeFiles(String file1, String file2)
			throws IOException {
		FileOutputStream fos = null;
		FileInputStream fis = null;

		try {
			fos = new FileOutputStream(file1, true);
			fis = new FileInputStream(file2);

			int read;
			byte[] b = new byte[1024];

			while ((read = fis.read(b)) > 0) {
				fos.write(b, 0, read);
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			if (fos != null) {
				fos.close();
			}
			if (fis != null) {
				fis.close();
			}
		}
	}

	/*
	 * Parses all the filtered blast2gene results and generates two files in
	 * order to calculate the overlappings
	 */
	public static void parseBl2G(String bl2gAnnot, String seqFile,
			String infile2) throws IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add("perl");
		params.add(BL2G_PARSE);
		params.add(bl2gAnnot);
		params.add(seqFile);
		params.add(infile2);

		System.out.println("Executing command:");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");

		}
		System.out.println();

		ProcessBuilder pb = new ProcessBuilder(params);
		ProcessHandler ph = new ProcessHandler(pb.start(), System.out,
				System.err);
		int exit = ph.waitFor();

		System.out.println("Exit status: " + exit);
	}

	/*
	 * Calculates the overlappings in different sequences
	 */
	public static void overlappingFromBL2G(String seqFile, String overlapFile)
			throws IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add("perl");
		params.add(OVERLAP);
		params.add(seqFile);
		params.add(seqFile);

		System.out.println("Executing command:");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");
		}
		System.out.println("> " + overlapFile);

		ProcessBuilder pb = new ProcessBuilder(params);
		FileOutputStream fos = new FileOutputStream(overlapFile);
		ProcessHandler ph = new ProcessHandler(pb.start(), fos, System.err);
		int exit = ph.waitFor();
		fos.close();

		System.out.println("Exit status: " + exit);
	}

	/*
	 * Formats the overlapping file in order to run genewise
	 */
	public static void overlappingFromBL2Gparse(String overlapFile,
			String infile2, String parsedOverlapFile) throws IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add("perl");
		params.add(OVERLAP_PARSE);
		params.add(overlapFile);
		params.add(infile2);

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");
		}
		System.out.println();

		ProcessBuilder pb = new ProcessBuilder(params);
		FileOutputStream fos = new FileOutputStream(parsedOverlapFile);
		ProcessHandler ph = new ProcessHandler(pb.start(), fos, System.err);
		int exit = ph.waitFor();
		fos.close();

		System.out.println("Exit status: " + exit);
	}

	/*
	 * Generates a file containing the subsequence of a genome delimited by
	 * start and end.
	 */
	public static void getSubsequence(String genomeFile, String start,
			String end, String subsequence) throws IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add("perl");
		params.add(GET_SUBSEQ);
		params.add(start);
		params.add(end);
		params.add(genomeFile);

		System.out.println("Executing command:");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");

		}
		System.out.println("> " + subsequence);

		ProcessBuilder pb = new ProcessBuilder(params);
		FileOutputStream fos = new FileOutputStream(subsequence);
		ProcessHandler ph = new ProcessHandler(pb.start(), fos, System.err);
		int exit = ph.waitFor();
		fos.close();
		System.out.println("Exit status: " + exit);
	}

	/*
	 * Runs genewise generating a report
	 */
	public static void runGenewise(String proteinFile, String subsequence,
			String strand, String genewiseReport) throws IOException {

		ArrayList<String> params = new ArrayList<String>();
		params.add(GENEWISE);
		params.add(proteinFile);
		params.add(subsequence);

		if ("Forward".equals(strand)) {
			params.add("-tfor");
		} else if ("Reverse".equals(strand)) {
			params.add("-trev");
		} else {
			params.add("-both");
		}
		params.add("-gff");
		params.add("-pretty");

		System.out.println("Executing command:");

		for (int i = 0; i < params.size(); i++) {
			System.out.print(params.get(i) + " ");
		}
		System.out.println("> " + genewiseReport);

		ProcessBuilder pb = new ProcessBuilder(params);
		Map<String, String> env = pb.environment();
		env.put("WISECONFIGDIR", WISECFG);
		FileOutputStream fos = new FileOutputStream(genewiseReport);
		ProcessHandler ph = new ProcessHandler(pb.start(), fos, null);
		int exit = ph.waitFor();
		fos.close();

		System.out.println("Exit status: " + exit);
	}

	public static Boolean generateZipFile(ArrayList<String> sourcesFilenames,
			String destinationDir, String zipFilename) {
		// Create a buffer for reading the files
		byte[] buf = new byte[1024];

		try {

			boolean result = (new File(destinationDir)).mkdirs();

			String zipFullFilename = destinationDir + "/" + zipFilename;

			// System.out.println(result);

			// Create the ZIP file
			ZipOutputStream out = new ZipOutputStream(new FileOutputStream(
					zipFullFilename));
			// Compress the files
			for (String filename : sourcesFilenames) {
				FileInputStream in = new FileInputStream(filename);
				// Add ZIP entry to output stream.
				File file = new File(filename);
				out.putNextEntry(new ZipEntry(file.getName()));
				// Transfer bytes from the file to the ZIP file
				int len;
				while ((len = in.read(buf)) > 0) {
					out.write(buf, 0, len);
				}
				// Complete the entry
				out.closeEntry();
				in.close();
			}

			// Complete the ZIP file
			out.close();

			return true;
		} catch (IOException e) {
			return false;
		}
	}

	public static void extractFolder(String zipFile, File destDir)
			throws ZipException, IOException {
		// System.out.println(zipFile);
		int BUFFER = 2048;
		File file = new File(zipFile);

		ZipFile zip = new ZipFile(file);
		String newPath = file.getCanonicalPath();
		// System.out.println(newPath);

		new File(newPath).mkdir();
		Enumeration zipFileEntries = zip.entries();
		try {
			// Process each entry
			while (zipFileEntries.hasMoreElements()) {
				// grab a zip file entry
				ZipEntry entry = (ZipEntry) zipFileEntries.nextElement();
				String currentEntry = entry.getName();
				// System.out.println(currentEntry);
				File destFile = new File(destDir.getPath(), currentEntry);

				// System.out.println(destFile);

				if (!entry.isDirectory()) {
					BufferedInputStream is = new BufferedInputStream(
							zip.getInputStream(entry));
					int currentByte;
					// establish buffer for writing file
					byte data[] = new byte[BUFFER];

					// write the current file to disk
					FileOutputStream fos = new FileOutputStream(destFile);
					BufferedOutputStream dest = new BufferedOutputStream(fos,
							BUFFER);

					// read and write until last byte is encountered
					while ((currentByte = is.read(data, 0, BUFFER)) != -1) {
						dest.write(data, 0, currentByte);
					}
					dest.flush();
					dest.close();
					is.close();
				}

				// if (currentEntry.endsWith(".zip")) {
				// // found a zip file, try to open
				// extractFolder(destFile.getAbsolutePath());
				// }
			}
		} finally {
			zip.close();
		}
	}
}
