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
package core.genedetect;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.util.Enumeration;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipOutputStream;

import core.genedetect.BL2GAnnotation;
import core.genedetect.BLASTText;
import core.genedetect.CommentedNASequence;
import core.genedetect.FASTA;
import core.genedetect.GFF;
import core.genedetect.NemusString;

import core.genedetect.GenewiseReport;

import core.genedetect.Filter;
import core.genedetect.RunNCBIBlastAgainstDBFromFASTASecondaryParameters;
import java.io.FileReader;

public class GeneDetectMethods {

    private static final String NAMESPACE = "ENSEMBL";
    //TODO change by the default optimis dir
    public static final String SERVICE_DIR = "/optimis_service/";
    private static final String PROTEIN_DB_DIR = "/ProteinDB/";
	private static final String ENCRYPTED_SPACE = "/optimis-sec-storage/";
    //private static final String SERVICE_DIR = System.getProperty("user.home") +"/servicess/";
    //private static final String SERVICE_DIR = System.getProperty("user.home") + "/ServiceSs/ServiceSs_TestBed/core";
    //private static final String SERVICE_DIR = System.getProperty("user.home") + "/ServiceSs/BIOMOBY";

    public static void runNCBIFormatdb(String genome, String genomeFile) {
        // Build formatdb command
        String formatdbBin = SERVICE_DIR + "/blast-2.2.15/bin/formatdb";
        String dataBase = SERVICE_DIR + "/genome/" + genome + ".fa";
        String commandLineArgs = "-p F"; // Type.DNA
        String cmd = formatdbBin + " -i " + dataBase + " -n " + genome + " " + commandLineArgs;

        // Run formatdb
        Process formatdbProc = null;
        try {
            printFileContent(dataBase);
            System.out.println("RUNNING COMMAND:\n" + cmd);
            int exitValue = 0;
            for (int i = 0; i < 3; i++) {
                System.out.println("Attempt " + i + " out of " + 3);
                formatdbProc = Runtime.getRuntime().exec(cmd);
                exitValue = formatdbProc.waitFor();
                System.out.println(exitValue);
                if (exitValue == 0) {
                    break;
                }
            }
            printOutputErrorStreams(formatdbProc.getErrorStream(), formatdbProc.getInputStream());

            if (exitValue != 0) {
                throw new Exception("Exit value is " + exitValue);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Zip the resulting files
        String zipFile = genomeFile;
        final int BUFFER = 2048;
        try {
            BufferedInputStream origin = null;
            FileOutputStream dest = new FileOutputStream(zipFile);
            ZipOutputStream out = new ZipOutputStream(new BufferedOutputStream(dest));
            byte data[] = new byte[BUFFER];
            File f = new File(".");
            String files[] = f.list();
            for (int i = 0; i < files.length; i++) {
                if (files[i].startsWith(genome + ".")
                        && (files[i].endsWith("nsq") || files[i].endsWith("nin") || files[i].endsWith("nhr"))) {
                    System.out.println("Adding: " + files[i]);
                    FileInputStream fi = new FileInputStream(files[i]);
                    origin = new BufferedInputStream(fi, BUFFER);
                    ZipEntry entry = new ZipEntry(files[i]);
                    out.putNextEntry(entry);
                    int count;
                    while ((count = origin.read(data, 0, BUFFER)) != -1) {
                        out.write(data, 0, count);
                    }
                    origin.close();
                    new File(files[i]).delete();
                }
            }
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        new File("formatdb.log").delete();
    }

    public static CommentedNASequence fromFastaToCommentedNASequence(String genome, String genomeFile) {
        // Build perl script command
        String dataBase = SERVICE_DIR + "/genome/" + genome + ".fa";
        String outfile = genomeFile; //"/tmp/outfile";
        String[] cmd = {"/bin/sh", "-c", SERVICE_DIR + "fromFastaToCommentedNASequence/FastaToCommNA.pl " + dataBase + " > " + outfile};

        // Run perl script
        Process perlProc = null;
        try {
            printFileContent(dataBase);
            System.out.println("RUNNING COMMAND:\n" + cmd[2]);

            int exitValue = 0;
            for (int i = 0; i < 10; i++) {
                System.out.println("Attempt " + i + " out of " + 3);
                perlProc = Runtime.getRuntime().exec(cmd);
                exitValue = perlProc.waitFor();
                System.out.println(exitValue);
                if (exitValue == 0) {
                    break;
                }
            }

            printOutputErrorStreams(perlProc.getErrorStream(), perlProc.getInputStream());
            printFileContent(outfile);
            if (exitValue != 0) {
                throw new Exception("Exit value is " + exitValue);
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        
        CommentedNASequence db = new CommentedNASequence();
        db.setNemusId(genome);
        db.setNemusNamespace(NAMESPACE);
        return db;
    }

    public static BLASTText runNCBIBlastp(FASTA fastaSeq, RunNCBIBlastpSecondaryParameters params) {
        // Put sequence string in a file
        File seqFile = null;
        File outFile = null;
        try {
            seqFile = File.createTempFile("seq", null);
            outFile = File.createTempFile("blastpOut", null);
            BufferedWriter out = new BufferedWriter(new FileWriter(seqFile));
            out.write('>' + fastaSeq.getNemusNamespace() + '|' + fastaSeq.getNemusId() + '\n');
            out.write(fastaSeq.getContent().getValue());
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        // Setup blastall parameters
        String blastallBin = SERVICE_DIR + "/blast-2.2.15/bin/blastall";
        String dataBase;
        if (params.getDatabase() == null) {
            dataBase = SERVICE_DIR + "/db_blast/nr.fasta.00";
        } else {
            dataBase = PROTEIN_DB_DIR+ params.getDatabase().value() + ".fasta";
        }

        String clineArgs = "-p blastp";
        if (params.getExtendgap() != null) {
            clineArgs += " -E " + params.getExtendgap();
        }
        if (params.getAlignments() != null) {
            clineArgs += " -b " + params.getAlignments();
        }
        if (params.getMatrix() != null) {
            clineArgs += " -M " + params.getMatrix().value();
        }
        if (params.getFilter() == Filter.TRUE) {
            clineArgs += " -F true";
        } else {
            clineArgs += " -F F";
        }
        if (params.getDropoff() != null) {
            clineArgs += " -X " + params.getDropoff();
        }
        if (params.getScores() != null) {
            clineArgs += " -v " + params.getScores();
        }
        if (params.getExpectedThreshold() != null) {
            clineArgs += " -e " + params.getExpectedThreshold();
        }
        if (params.getOpengap() != null) {
            clineArgs += " -G " + params.getOpengap();
        }
        if (params.getGapalign() != null) {
            clineArgs += " -g " + params.getGapalign().value();
        }

        String cmd = blastallBin + " -i " + seqFile.getAbsolutePath() + " -d " + dataBase + " " + clineArgs + " -a 1 -o " + outFile.getAbsolutePath();

        // Run blastall
        Process blastallProc = null;
        try {
            int exitValue = 0;
            for (int i = 0; i < 10; i++) {
                System.out.println("Attempt " + i + " out of " + 3);
                blastallProc = Runtime.getRuntime().exec(cmd);
                exitValue = blastallProc.waitFor();
                System.out.println(exitValue);
                if (exitValue == 0) {
                    break;
                }
            }

            printOutputErrorStreams(blastallProc.getErrorStream(), blastallProc.getInputStream());
            printFileContent(outFile.getAbsolutePath());
            if (exitValue != 0) {
                throw new Exception("Exit value is " + exitValue);
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        // Read blastall output file
        StringBuilder writer = new StringBuilder();
        char[] buffer = new char[1024];
        try {
            InputStream fis = new FileInputStream(outFile);
            Reader reader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            int n;
            while ((n = reader.read(buffer)) != -1) {
                writer.append(buffer, 0, n);
            }
            fis.close();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
        String blastpReport = writer.toString();

        // Create response object with report
        BLASTText blastReport = new BLASTText();
        NemusString content = new NemusString();
        content.setNemusId(fastaSeq.getNemusId());
        content.setNemusNamespace(NAMESPACE);
        content.setValue(blastpReport);
        blastReport.setContent(content);

        seqFile.delete();
        outFile.delete();

        return blastReport;
    }

    public static BLASTText runNCBIBlastAgainstDBFromFASTA(
            String blastDB,
            FASTA fasta,
            RunNCBIBlastAgainstDBFromFASTASecondaryParameters params) {

        String tempDbFile = blastDB;

        // Unzip db file
        String dbName = null;
        ZipFile zipfile = null;
        try {
            final int BUFFER = 2048;
            BufferedOutputStream dest = null;
            BufferedInputStream is = null;
            ZipEntry entry;
            zipfile = new ZipFile(tempDbFile);
            Enumeration<? extends ZipEntry> e = zipfile.entries();
            while (e.hasMoreElements()) {
                entry = (ZipEntry) e.nextElement();
                String fileName = entry.getName();
                if (fileName.endsWith(".nin")) {
                    dbName = fileName.substring(0, fileName.indexOf(".nin"));
                }
                is = new BufferedInputStream(zipfile.getInputStream(entry));
                int count;
                byte data[] = new byte[BUFFER];
                FileOutputStream fos = new FileOutputStream(entry.getName());
                System.out.println("File Name: "+ entry.getName() );
                dest = new BufferedOutputStream(fos, BUFFER);
                while ((count = is.read(data, 0, BUFFER)) != -1) {
                    dest.write(data, 0, count);
                }
                dest.flush();
                dest.close();
                is.close();
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        } finally {
        	try {
				zipfile.close();
			} catch (IOException e) {
				 e.printStackTrace();
		         System.exit(1);
			}
        }

        // Put sequence in a file
        File tempSeqFile = null;
        File outFile = null;
        try {
            tempSeqFile = File.createTempFile("seq", null);
            outFile = File.createTempFile("blastOut", null);
            OutputStream bos = new BufferedOutputStream(new FileOutputStream(tempSeqFile));
            bos.write(fasta.getContent().getValue().getBytes());
            bos.close();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        // Setup blastall parameters
        String blastallBin = SERVICE_DIR + "/blast-2.2.15/bin/blastall";
        String clineArgs = "";
        if (params.getExtendgap() != null) {
            clineArgs += " -E " + params.getExtendgap();
        }
        if (params.getAlignments() != null) {
            clineArgs += " -b " + params.getAlignments();
        }
        if (params.getMatrix() != null) {
            clineArgs += " -M " + params.getMatrix().value();
        }
        if (params.getFilter() == Filter.TRUE) {
            clineArgs += " -F true";
        }
        if (params.getDropoff() != null) {
            clineArgs += " -X " + params.getDropoff();
        }
        if (params.getScores() != null) {
            clineArgs += " -v " + params.getScores();
        }
        if (params.getExpectedThreshold() != null) {
            clineArgs += " -e " + params.getExpectedThreshold();
        }
        if (params.getOpengap() != null) {
            clineArgs += " -G " + params.getOpengap();
        }
        if (params.getGapalign() != null) {
            clineArgs += " -g " + params.getGapalign().value();
        }
        if (params.getProgram() != null) {
            clineArgs += " -p " + params.getProgram().value();
        }

        String cmd = blastallBin + " -i " + tempSeqFile.getAbsolutePath() + " -d " + dbName + clineArgs + " -a 1 -o " + outFile.getAbsolutePath();

        // Run blastall
        Process blastallProc = null;
        try {
            printFileContent(tempSeqFile.getAbsolutePath());
            System.out.println("RUNNING COMMAND:\n" + cmd);
            int exitValue = 0;
            for (int i = 0; i < 10; i++) {
                System.out.println("Attempt " + i + " out of " + 3);
                blastallProc = Runtime.getRuntime().exec(cmd);
                exitValue = blastallProc.waitFor();
                System.out.println(exitValue);
                if (exitValue == 0) {
                    break;
                }
            }
            printOutputErrorStreams(blastallProc.getErrorStream(), blastallProc.getInputStream());
            printFileContent(outFile.getAbsolutePath());
            if (exitValue != 0) {
                throw new Exception("Exit value is " + exitValue);
            }

        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Read blastall output file
        StringBuilder writer = new StringBuilder();
        char[] buffer = new char[1024];
        try {
            InputStream fis = new FileInputStream(outFile);
            Reader reader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            int n;
            while ((n = reader.read(buffer)) != -1) {
                writer.append(buffer, 0, n);
            }
            fis.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        String tblastnReport = writer.toString();

        // Create response object
        BLASTText blastReport = new BLASTText();
        NemusString content = new NemusString();
        content.setNemusNamespace(NAMESPACE);
        content.setValue(tblastnReport);
        blastReport.setContent(content);

        tempSeqFile.delete();
        outFile.delete();

        return blastReport;
    }

    public static BLASTText mergeBlastResults(BLASTText res1, BLASTText res2) {
        if (res1 == null && res2 != null){
        	System.err.println("Blast result 1 is null, returning result 2");
        	return res1;
        } else if (res1 != null && res2 == null){
        	System.err.println("Blast result 2 is null, returning result 1");
        	return res2;
        } else if (res1 == null && res2 == null){
        	System.err.println("Both results are null, returning null");
        	return res1;
        }
    	String content1 = res1.getContent().getValue();
        String content2 = res2.getContent().getValue();
        
        BLASTText mergedReport = new BLASTText();

        NemusString concatNemus = new NemusString();
        concatNemus.setValue(content1 + "\n" + content2);

        mergedReport.setContent(concatNemus);
        mergedReport.setNemusId(res1.getNemusId());
        mergedReport.setNemusNamespace(res1.getNemusNamespace());

        return mergedReport;
    }

    
    
    public static String prepareGenewiseFiles(String genomeCNAFile, CommentedNASequence genomeDB, BL2GAnnotation region, FASTA sequence, String seqFile, String genomeFile){
    	// Parse region
        int start = region.getStart().getValue();
        int end = region.getEnd().getValue();
        String strand = region.getStrand().getValue();
        try {
			Thread.sleep(17000);
		} catch (InterruptedException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
        // Cut genome
        StringBuilder writer = new StringBuilder();
        char[] buffer = new char[1024];
        try {
            InputStream fis = new FileInputStream(genomeCNAFile);
            Reader reader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            int n;
            while ((n = reader.read(buffer)) != -1) {
                writer.append(buffer, 0, n);
            }
            fis.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        String genomeString = writer.toString();
        String cutGenomeString = getSubSequence(genomeString, start - 5000, end + 5000);

        // Put sequence string in a file
        String sequenceString = sequence.getContent().getValue();
        //sequenceString = sequenceString.replaceAll("\\s", "");
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(seqFile));
            //out.write('>' + sequence.getNemusNamespace() + '|' + sequence.getNemusId() + '\n');
            out.write(sequenceString);
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Put genome string in a file
        try {
            BufferedWriter out = new BufferedWriter(new FileWriter(genomeFile));
            out.write('>' + genomeDB.getNemusNamespace() + '|' + genomeDB.getNemusId() + '\n');
            out.write(cutGenomeString);
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Setup genewise parameters
        String commandLineArgs = "";
        if (strand.equals("Forward")) {
            commandLineArgs += " -tfor ";
        } else if (strand.equals("Reverse")) {
            commandLineArgs += " -trev ";
        } else {
            commandLineArgs += " -both ";
        }
        return commandLineArgs;
    }
    
    /*public static String runGenewiseBinary(String seqFile,
            String genomeFile,
            String args) {
    	return null;
    }*/
    
    public static GenewiseReport postProcessGenewise(String rawReport, BL2GAnnotation region){
    	int start = region.getStart().getValue();
    	GFF report = new GFF();
    	try {
    		String postProcessedReport = readjustGWReport(new ByteArrayInputStream(rawReport.getBytes("UTF-8")), start);
    		NemusString content = new NemusString();
    		content.setValue(postProcessedReport);
    		report.setContent(content);

         
    	} catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
    	GenewiseReport gwReport = new GenewiseReport(report);
        return gwReport;
    
    }
    public static GenewiseReport runGenewise(String genomeCNAFile,
            CommentedNASequence genomeDB,
            BL2GAnnotation region,
            FASTA sequence) {
        // Parse region
        int start = region.getStart().getValue();
        int end = region.getEnd().getValue();
        String strand = region.getStrand().getValue();

        // Cut genome
        StringBuilder writer = new StringBuilder();
        char[] buffer = new char[1024];
        try {
            InputStream fis = new FileInputStream(genomeCNAFile);
            Reader reader = new BufferedReader(new InputStreamReader(fis, "UTF-8"));
            int n;
            while ((n = reader.read(buffer)) != -1) {
                writer.append(buffer, 0, n);
            }
            fis.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }
        String genomeString = writer.toString();
        String cutGenomeString = getSubSequence(genomeString, start - 5000, end + 5000);

        // Put sequence string in a file
        String sequenceString = sequence.getContent().getValue();
        //sequenceString = sequenceString.replaceAll("\\s", "");
        File seqFile = null;
        try {
            seqFile = File.createTempFile("seq", null);
            BufferedWriter out = new BufferedWriter(new FileWriter(seqFile));
            //out.write('>' + sequence.getNemusNamespace() + '|' + sequence.getNemusId() + '\n');
            out.write(sequenceString);
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Put genome string in a file
        File genomeFile = null;
        try {
            genomeFile = File.createTempFile("genome", null);
            BufferedWriter out = new BufferedWriter(new FileWriter(genomeFile));
            out.write('>' + genomeDB.getNemusNamespace() + '|' + genomeDB.getNemusId() + '\n');
            out.write(cutGenomeString);
            out.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Setup genewise parameters
        String commandLineArgs = "";
        if (strand.equals("Forward")) {
            commandLineArgs += " -tfor ";
        } else if (strand.equals("Reverse")) {
            commandLineArgs += " -trev ";
        } else {
            commandLineArgs += " -both ";
        }
        String genewiseBin = SERVICE_DIR + "/wise2.2.0/src/bin/genewise";
        String[] cmd = {"/bin/sh", "-c", genewiseBin + " " + seqFile.getAbsolutePath() + " " + genomeFile.getAbsolutePath() + " " + commandLineArgs + " -gff"};

        // Run genewise
        ProcessBuilder pb = new ProcessBuilder(cmd);
        Map<String, String> env = pb.environment();
        env.put("WISECONFIGDIR", "/optimis_service/wise2.2.0/wisecfg/");
        Process genewiseProc = null;
        String postProcessedReport = null;
        try {
            System.out.println("RUNNING COMMAND:\n" + cmd[2]);

            int exitValue = 0;

            for (int i = 0; i < 3; i++) {
                System.out.println("Attempt " + i + " out of " + 3);
                genewiseProc = pb.start();

                // Read error to prevent the subprocess from blocking
                BufferedInputStream bisErr = new BufferedInputStream(genewiseProc.getErrorStream());
                byte[] b = new byte[1024];
                while ((bisErr.read(b)) >= 0) {
                }
                bisErr.close();
                genewiseProc.getErrorStream().close();
                genewiseProc.getOutputStream().close();

                // Read genewise process output and readjust the report
                postProcessedReport = readjustGWReport(genewiseProc.getInputStream(), start);

                exitValue = genewiseProc.waitFor();

                genewiseProc.destroy();
                if (exitValue == 0) {
                    break;
                }
            }
            if (exitValue != 0) {
                throw new Exception("Exit value is " + exitValue);
            }











        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        // Create response object
        GFF report = new GFF();
        NemusString content = new NemusString();
        content.setValue(postProcessedReport);
        report.setContent(content);

        GenewiseReport gwReport = new GenewiseReport(report);

        return gwReport;
    }

    private static String getSubSequence(String genome, int ini, int end) {
        int count = end - ini;
        int start;

        int rc = 0;
        String seq = "";

        boolean finish = false;
        String[] arr = genome.split("\n");

        for (String line : arr) {
            int l = line.length();
            if (l + rc < ini) {
                rc += l;
            } else {
                if (ini > rc && ini < rc + l) {
                    start = ini - rc;
                } else {
                    start = 0;
                }
                if (end >= l + rc) {
                    seq += line.substring(start, start + l);
                } else /*if (end < l + rc) */ {
                    seq += line.substring(start, start + (count - rc));
                    finish = true;
                }
                rc += l;
            }

            if (finish) {
                break;
            }
        }

        return seq;
    }

    private static String readjustGWReport(InputStream fis, int start) {
        String report = "";
        try {
            BufferedReader br = new BufferedReader(new InputStreamReader(fis));
            String line;
            Pattern p = Pattern.compile("^(.+\\s+\\S+\\s+)(\\d+)\\s+(\\d+)(\\s+.+)");
            while ((line = br.readLine()) != null) {
                Matcher m = p.matcher(line);
                if (m.find()) {
                    int ini = Integer.parseInt(m.group(2)) + start - 5000;
                    int end = Integer.parseInt(m.group(3)) + start - 5000;
                    if (ini > end) {
                        report += m.group(1) + end + "\t" + ini + m.group(4) + "\n\n";
                    } else {
                        report += m.group(1) + ini + "\t" + end + m.group(4) + "\n\n";
                    }
                }
            }
            fis.close();
            br.close();
        } catch (Exception e) {
            e.printStackTrace();
            System.exit(1);
        }

        return report;
    }

    private static void printOutputErrorStreams(InputStream errorStream, InputStream inputStream) {
        int bytesRead = 0;
        byte[] buffer = new byte[1024];
        BufferedInputStream bis;
        try {
            bis = new BufferedInputStream(errorStream);
            //Keep reading from the file while there is any content
            //when the end of the stream has been reached, -1 is returned
            while ((bytesRead = bis.read(buffer)) != -1) {
                //Process the chunk of bytes read
                //in this case we just construct a String and print it out
                String chunk = new String(buffer, 0, bytesRead);
                System.err.print(chunk);
            }
            bis.close();
        } catch (Exception e) {
            System.err.println("Error when printing the error stream");
        }
        try {
            bis = new BufferedInputStream(inputStream);
            //Keep reading from the file while there is any content
            //when the end of the stream has been reached, -1 is returned
            while ((bytesRead = bis.read(buffer)) != -1) {
                //Process the chunk of bytes read
                //in this case we just construct a String and print it out
                String chunk = new String(buffer, 0, bytesRead);
                System.out.print(chunk);
            }
            bis.close();
        } catch (Exception e) {

            System.err.println("Error when printing the output stream");

        }
    }

    private static void printFileContent(String filePath) {
    	BufferedReader in = null;
        try {
            System.out.println(filePath + " start");
            in = new BufferedReader(new FileReader(filePath));
            String str;
            while ((str = in.readLine()) != null) {
                System.out.println(str);
            }
            System.out.println(filePath + " end");
        } catch (Exception e) {
            System.err.println("Error reading the file " + filePath);
            System.out.println("Error reading the file " + filePath);
            e.printStackTrace();
        } finally {
        	if (in != null) {
        		try {
					in.close();
				} catch (IOException e) {
					System.err.println("Error closing the file " + filePath);
				}
        	}
        }
    }

	public static void bindSequenceToSecureData(FASTA fastaSeq) {
		// TODO When secure storage is ready
		BufferedWriter out;
		try {
			out = new BufferedWriter(new FileWriter(ENCRYPTED_SPACE+"/"+fastaSeq.getNemusId()+".fasta"));
			out.write(fastaSeq.getContent().getValue());
			out.close();
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static void loadDataBase(){
		// TODO When secure storage is ready
	}

	public static void removeSecureSequence(String sequenceName) {
		// TODO When secure storage is ready
		File f = new File(ENCRYPTED_SPACE+"/"+sequenceName+".fasta");
		f.delete();
	}

	public static void bindResultsToSecureData(GenewiseReport report, String runID) {
		BufferedWriter out;
		try {
			out = new BufferedWriter(new FileWriter(ENCRYPTED_SPACE+"/"+runID+"_report.gff"));
			out.write(report.getGff().getContent().getValue());
			out.close();
        } catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}

	public static void removeResultFromSecureData(String runID) {
		// TODO Auto-generated method stub
		File f = new File(ENCRYPTED_SPACE+"/"+runID+"_report.gff");
		f.delete();
	}

}
