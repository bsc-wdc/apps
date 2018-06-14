package core.genedetect;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
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

import core.genedetect.AminoAcidSequence;
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
    private static final String SERVICE_DIR = "/optimis_service/";
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
            for (int i = 0; i < 3; i++) {
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
            dataBase = "/ProteinDB/" + params.getDatabase().value() + ".fasta";
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
            for (int i = 0; i < 3; i++) {
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
        try {
            final int BUFFER = 2048;
            BufferedOutputStream dest = null;
            BufferedInputStream is = null;
            ZipEntry entry;
            ZipFile zipfile = new ZipFile(tempDbFile);
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
            for (int i = 0; i < 3; i++) {
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
        env.put("WISECONFIGDIR", SERVICE_DIR + "/wise2.2.0/wisecfg/");
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
                postProcessedReport = readjustGWReport(genewiseProc, start);

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

    private static String readjustGWReport(Process genewiseProc, int start) {
        String report = "";
        try {
            InputStream fis = genewiseProc.getInputStream();
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
        try {
            System.out.println(filePath + " start");
            BufferedReader in = new BufferedReader(new FileReader(filePath));
            String str;
            while ((str = in.readLine()) != null) {
                System.out.println(str);
            }
            System.out.println(filePath + " end");
        } catch (Exception e) {
            System.err.println("Error reading the file " + filePath);
            System.out.println("Error reading the file " + filePath);
            e.printStackTrace();
        }
    }
}