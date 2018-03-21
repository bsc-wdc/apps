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
package sequential.genedetection;

import static worker.genedetection.GeneDetectionImpl.filterBl2G;
import static worker.genedetection.GeneDetectionImpl.getSubsequence;
import static worker.genedetection.GeneDetectionImpl.mergeFiles;
import static worker.genedetection.GeneDetectionImpl.overlappingFromBL2G;
import static worker.genedetection.GeneDetectionImpl.overlappingFromBL2Gparse;
import static worker.genedetection.GeneDetectionImpl.parseBl2G;
import static worker.genedetection.GeneDetectionImpl.runBL2Gparse;
import static worker.genedetection.GeneDetectionImpl.runBlast2Gene;
import static worker.genedetection.GeneDetectionImpl.runGenewise;
import static worker.genedetection.GeneDetectionImpl.runNCBIFormatdb;
import static worker.genedetection.GeneDetectionImpl.runNCBIBlastAgainstDB;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.Queue;

import org.biojava3.core.sequence.ProteinSequence;
import org.biojava3.core.sequence.io.FastaReaderHelper;

/**
 * 
 * @author amantsok
 */
public class GeneDetection {

	private static final String DB_TYPE = "nucl";
	private static final String FILE_TYPE = "fasta";
	private static final String DB_PATH = "/blastdb";
	private static final String DB_TITLE = "blastdb";

	public static void main(String[] args) throws IOException, Exception {

		// Arguments:
		// 1- Path to genome file
		// 2- Path to proteins file
		// 3- Output directory
		String genomePath = args[0];
		File proteinsFile = new File(args[1]);
		File resultsDir = new File(args[2]);
		resultsDir.mkdir();

		File zipFile = new File(resultsDir + "/blastdb.zip");
		File protFolder = new File(resultsDir + "/proteins");
		protFolder.mkdir();

		/* ***********************
		 * 
		 * A. Generate Genome DB
		 * 
		 * ***********************
		 */

		runNCBIFormatdb(genomePath, DB_TYPE, FILE_TYPE, DB_PATH, DB_TITLE, zipFile.getPath());

		/* *****************************************
		 * 
		 * B. Find relevant genes (with blast2gene)
		 * 
		 * *****************************************
		 */

		LinkedHashMap<String, ProteinSequence> proteins = FastaReaderHelper.readFastaProteinSequence(proteinsFile);
		Queue<File> mergeList = new LinkedList<File>();

		for (ProteinSequence protein : proteins.values()) {
			String proteinPath = proteinToFile(protein, protFolder);
			File blastReport = File.createTempFile("blast", null, resultsDir);
			File parseOut = File.createTempFile("parsed", ".bp", resultsDir);
			File bl2gFile = File.createTempFile("bl2g", null, resultsDir);
			File bl2gFilter = File.createTempFile("bl2gFilter", null, resultsDir);

			// run blast
			runNCBIBlastAgainstDB(proteinPath, DB_PATH, zipFile.getPath(), blastReport.getPath());

			// parse blast report
			runBL2Gparse(blastReport.getPath(), parseOut.getPath());

			// run blast2gene
			runBlast2Gene(parseOut.getPath(), bl2gFile.getPath());

			// filter blast2gene results
			filterBl2G(bl2gFile.getPath(), bl2gFilter.getPath());

			mergeList.add(bl2gFilter);
		}
		// merge all blast2gene results into a single file
		while (mergeList.size() > 1) {
			File f1 = mergeList.poll();
			File f2 = mergeList.poll();

			mergeFiles(f1.getPath(), f2.getPath());

			mergeList.add(f1);
		}
		File bl2gAnnotations = mergeList.poll();
		File seqFasta = File.createTempFile("seq", ".fasta", resultsDir);
		File infile2 = File.createTempFile("infile2", null, resultsDir);
		File overlappings = File.createTempFile("overlap", null, resultsDir);
		File parsedOverlap = File.createTempFile("overlapParsed", null, resultsDir);

		// parse blast2gene results
		parseBl2G(bl2gAnnotations.getPath(), seqFasta.getPath(), infile2.getPath());

		// calculate overlappings
		overlappingFromBL2G(seqFasta.getPath(), overlappings.getPath());

		// parse overlapping results
		overlappingFromBL2Gparse(overlappings.getPath(), infile2.getPath(), parsedOverlap.getPath());

		/* *****************************************
		 * 
		 * C. Generate GFF report
		 * 
		 * *****************************************
		 */

		int i = 1;
		BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(parsedOverlap)));
		String line;

		while ((line = br.readLine()) != null) {
			File subseqFile = new File(resultsDir + "/subseq" + i + ".fasta");
			File genewiseRep = new File(resultsDir + "/genewise_rep" + i);

			String[] tokens = line.split("\\s+");
			String start = tokens[0];
			String end = tokens[1];
			String strand = tokens[2];
			String id = tokens[3];

			// get subsequence
			getSubsequence(genomePath, start, end, subseqFile.getPath());

			// run genewise
			for (File file : protFolder.listFiles()) {
				if (file.getName().startsWith(id)) {
					runGenewise(file.getPath(), subseqFile.getPath(), strand, genewiseRep.getPath());
				}
			}
			i++;
		}
	}

	private static String proteinToFile(ProteinSequence protein, File directory) throws IOException {
		String protId = protein.getAccession().toString();
		String proteinPath = directory.getPath() + "/" + protId + ".fasta";
		FileOutputStream writer = new FileOutputStream(proteinPath);
		String tmp = ">" + protId + "\n" + protein.getSequenceAsString();
		writer.write(tmp.getBytes());
		writer.close();

		return proteinPath;
	}
}
