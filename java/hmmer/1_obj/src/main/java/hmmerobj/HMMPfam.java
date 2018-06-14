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
package hmmerobj;

import java.io.BufferedInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.channels.FileChannel;
import java.util.LinkedList;


public class HMMPfam {

    private static final String SEQUENCE = "seqF";
    private static final String DATABASE = "dbF";
    private static final String FRAGS_DIR = "/tmp/hmmer_frags";
    // hmmpfam constants
    private static final int ALPHABET_SIZE = 20;
    private static final int TRANS_PROB_SIZE = 7;
    private static final int PLAN7_DESC = 1 << 1;
    private static final int PLAN7_RF = 1 << 2;
    private static final int PLAN7_CS = 1 << 3;
    private static final int PLAN7_STATS = 1 << 7;
    private static final int PLAN7_MAP = 1 << 8;
    private static final int PLAN7_ACC = 1 << 9;
    private static final int PLAN7_GA = 1 << 10;
    private static final int PLAN7_TC = 1 << 11;
    private static final int PLAN7_NC = 1 << 12;
    // Let's assume these are the right sizes for file reading (in bytes)
    private static final int CHAR_SIZE = 1;
    private static final int INT_SIZE = 4;
    private static final int FLOAT_SIZE = 4;
    private static final byte[] intBytes = new byte[INT_SIZE];
    private static final byte iniSeq = 0x3e;
    private static final int v20Magic = 0xe8ededb5;

    private static final String fragsDirName;
    private static final String hmmpfamBin;

    private static CommandLineArgs clArgs;
    private static int[] dbFragsNumModels = null;
    private static int totalNumModels = 0;
    private static boolean debug = false;

    static {
        String fDirTmp = "";
        File fragsDir = new File(FRAGS_DIR);
        if (!fragsDir.exists() && !fragsDir.mkdir()) {
            System.out.println("Cannot create the fragments directory");
            System.exit(1);
        }
        try {
            fDirTmp = fragsDir.getCanonicalPath();
        } catch (Exception e) {
            System.out.println("Cannot get the name of the fragments directory");
            e.printStackTrace();
            System.exit(1);
        }
        fragsDirName = fDirTmp;
        fragsDir.deleteOnExit();

        String binPath = System.getenv("HMMER_BINARY");
        if (binPath == null) {
            binPath = "/usr/local/bin/hmmpfam";
        }
        hmmpfamBin = binPath;
    }

    public static void main(String args[]) throws Exception {
        String dbName = args[0];
        String seqsName = args[1];
        String outputName = args[2];

        int numDBFrags = Integer.parseInt(args[3]);
        int numSeqFrags = Integer.parseInt(args[4]);

        clArgs = new CommandLineArgs(args, 5);

        File fSeq = new File(seqsName);
        File fDB = new File(dbName);

        LinkedList<String> seqFrags = new LinkedList<String>();
        LinkedList<String> dbFrags = new LinkedList<String>();

        if (debug) {
            System.out.println("\nParameters: ");
            System.out.println("- Database file: " + dbName);
            System.out.println("- Query sequences file: " + seqsName);
            System.out.println("- Output file: " + outputName);
            System.out.println("- Command line args: " + clArgs.getArgs());
            System.out.println("- Number of db frags: " + numDBFrags);
            System.out.println("- Number of seq frags: " + numSeqFrags);
            System.out.println("- Frags dir: " + fragsDirName);
            System.out.println("- hmmpfam binary: " + hmmpfamBin);
        }

        /* FIRST PHASE
         * Segment the query sequences file, the database file or both
         */
        split(fSeq, fDB, seqFrags, dbFrags, numSeqFrags, numDBFrags);


        /* SECOND PHASE
         * Launch hmmpfam for each pair of seq - db fragments
         */
        numSeqFrags = seqFrags.size();
        numDBFrags = dbFrags.size();
        String commonArgs = clArgs.getArgs();

        int i = 0;
        int startHMM = 0;
        int dbNum = 0;
        String[][] outputs = new String[numDBFrags][numSeqFrags];

        for (String dbFrag : dbFrags) {
            String finalArgs = commonArgs;
            if (dbFragsNumModels != null) {
                startHMM += dbFragsNumModels[i++];
            }
            int seqNum = 0;
            for (String seqFrag : seqFrags) {
                outputs[dbNum][seqNum] = HMMPfamImpl.hmmpfam(hmmpfamBin,
                        finalArgs,
                        seqFrag,
                        dbFrag);
                seqNum++;
            }
            dbNum++;
        }


        /* THIRD PHASE
         * Merge all output in a single file
         */
        // Merge all DB outputs for the same DB fragment
        for (int db = 0; db < numDBFrags; db++) {
            int neighbor = 1;
            String[] sequences = outputs[db];
            while (neighbor < numSeqFrags) {
                for (int seq = 0; seq < numSeqFrags; seq += 2 * neighbor) {
                    if (seq + neighbor < numSeqFrags) {
                        sequences[seq] = HMMPfamImpl.mergeSameDB(sequences[seq], sequences[seq + neighbor]);
                    }
                }
                neighbor *= 2;
            }
        }

        // Merge all outputs for the whole DB
        int neighbor = 1;
        while (neighbor < numDBFrags) {
            for (int db = 0; db < numDBFrags; db += 2 * neighbor) {
                if (db + neighbor < numDBFrags) {
                    outputs[db][0] = HMMPfamImpl.mergeSameSeq(outputs[db][0], outputs[db + neighbor][0], clArgs.getALimit());
                }
            }
            neighbor *= 2;
        }

        try {
            prepareResultFile(outputs[0][0], outputName, seqsName, dbName);
        } catch (IOException e) {
            System.out.println("Error copying final result file");
            e.printStackTrace();
            System.exit(1);
        }

        // Clean
        if (seqFrags.size() > 1) {
            for (String seqFrag : seqFrags) {
                new File(seqFrag).delete();
            }
        }
        if (dbFrags.size() > 1) {
            for (String dbFrag : dbFrags) {
                new File(dbFrag).delete();
            }
        }
    }

    // Split the query sequences file, the database file or both
    private static void split(File fSeq,
            File fDB,
            LinkedList<String> seqFrags,
            LinkedList<String> dbFrags,
            int numSeqFrags,
            int numDBFrags) {

        long fSeqSize = fSeq.length(), // B
                fDBSize = fDB.length();  // B

        long seqFragSize = fSeqSize / numSeqFrags;
        long dbFragSize = fDBSize / numDBFrags;

        long timeStamp = Thread.currentThread().getId();
        //long timeStamp = System.nanoTime();

        if (debug) {
            System.out.println("\nDecided sizes for fragments:");
            System.out.println("- Sequences file size (B): " + fSeqSize);
            System.out.println("- Sequences fragment size (B): " + seqFragSize);
            System.out.println("- DB file size (B): " + fDBSize);
            System.out.println("- DB fragment size (B): " + dbFragSize);
        }

        // Now generate the fragments (if necessary)
        try {
            if (fSeqSize == seqFragSize) {
                seqFrags.add(fSeq.getCanonicalPath());
            } else {
                generateSeqFragments(fSeq, seqFragSize, seqFrags, timeStamp);
            }

            if (fDBSize == dbFragSize) {
                dbFrags.add(fDB.getCanonicalPath());
            } else {
                generateDBFragments(fDB, dbFragSize, dbFrags, timeStamp);
            }
        } catch (Exception e) {
            System.out.println("Error generating fragments");
            e.printStackTrace();
            System.exit(1);
        }
    }

    /* Visit the query sequence file iteratively, skipping fragSize bytes and then
     * searching for '>' to determine the end of the fragment at each iteration.
     */
    private static void generateSeqFragments(File srcFile,
            long fragSize,
            LinkedList<String> frags,
            long timeStamp) throws Exception {

        if (debug) {
            System.out.println("\nGenerating sequence fragments");
        }
        RandomAccessFile raf = new RandomAccessFile(srcFile, "r");

        FileChannel seqChannel = raf.getChannel();

        // Position of the first byte in the fragment
        long iniFragPosition = 0;
        int fragNum = 0;
        boolean eofReached = false;

        while (!eofReached) {
            seqChannel.position(iniFragPosition + fragSize);

            boolean fragRead = false;
            while (!fragRead) {
                try {
                    fragRead = (raf.readByte() == iniSeq);
                } catch (EOFException e) {
                    eofReached = true;
                    fragRead = true;
                    continue;
                }
            }

            // Position right after the last byte of the fragment (position of token)
            long endFragPosition = eofReached ? seqChannel.size() : seqChannel.position() - 1;

            // Channel for the current fragment
            String fragPathName = fragsDirName + File.separator + SEQUENCE + fragNum + "_" + timeStamp;
            FileChannel fragChannel = new FileOutputStream(fragPathName).getChannel();
            frags.add(fragPathName);

            if (debug) {
                System.out.println("\t- Fragment " + fragNum + ": transfer from " + srcFile.getCanonicalPath() + " to " + fragPathName
                        + ", " + (endFragPosition - iniFragPosition) + " bytes starting at byte " + iniFragPosition);
            }

            seqChannel.transferTo(iniFragPosition, endFragPosition - iniFragPosition, fragChannel);
            iniFragPosition = endFragPosition;
            fragNum++;
        }

        raf.close();
    }

    // Visit the DB file, reading one model at each iteration.
    private static void generateDBFragments(File srcFile,
            long fragSize,
            LinkedList<String> frags,
            long timeStamp) throws Exception {

        if (debug) {
            System.out.println("\nGenerating database fragments");
        }

        BufferedInputStream bisDB = new BufferedInputStream(new FileInputStream(srcFile));
        bisDB.mark(5);
        final boolean isSwap = (readInt(bisDB) == Integer.reverseBytes(v20Magic));
        bisDB.reset();

        FileChannel dbChannel = new FileInputStream(srcFile).getChannel();
        int numFrags = (int) (srcFile.length() / fragSize);
        int toAdd = (int) (srcFile.length() % fragSize);
        dbFragsNumModels = new int[numFrags];
        int fragNum = 0;
        int fragNumModels = 0;
        long fragBytes = 0;
        long iniFrag = 0;
        long modelBytes;

        while ((modelBytes = readHMM(bisDB, isSwap)) > 0) {
            fragNumModels++;
            fragBytes += modelBytes;
            if (fragBytes >= fragSize) {
                // Channel for the current fragment
                String fragPathName = fragsDirName + File.separator + DATABASE + fragNum + "_" + timeStamp;
                FileChannel fragChannel = new FileOutputStream(fragPathName).getChannel();
                //fragPathName = "shared://shared1/hmmer_frags" + File.separator + DATABASE + fragNum + "_" + timeStamp;
                frags.add(fragPathName);

                if (debug) {
                    System.out.println("\t- Fragment " + fragNum + ": transfer from " + srcFile.getCanonicalPath() + " to " + fragPathName
                            + ", " + fragBytes + " bytes starting at byte " + iniFrag);
                }

                dbChannel.transferTo(iniFrag, fragBytes, fragChannel);

                iniFrag += fragBytes;
                dbFragsNumModels[fragNum] = fragNumModels;
                totalNumModels += fragNumModels;
                fragNumModels = 0;
                fragBytes = 0;
                fragNum++;

                // Make sure that last fragment has the remainder of bytes
                if (fragNum == numFrags - 1) {
                    fragSize += toAdd;
                }
            }
        }
        // Treat last fragment
        if (fragBytes > 0) {
            String fragPathName = fragsDirName + File.separator + DATABASE + fragNum;
            FileChannel fragChannel = new FileOutputStream(fragPathName).getChannel();
            frags.add(fragPathName);
            if (debug) {
                System.out.println("\t- Fragment " + fragNum + ": transfer from " + srcFile.getCanonicalPath() + " to " + fragPathName
                        + ", " + fragBytes + " bytes starting at byte " + iniFrag);
            }
            dbChannel.transferTo(iniFrag, fragBytes, fragChannel);

            dbFragsNumModels[fragNum] = fragNumModels;
            totalNumModels += fragNumModels;
        } else {
            fragNum--;
        }

        bisDB.close();
        dbChannel.close();

        if (debug) {
            System.out.println("Found a total of " + totalNumModels + " models in the database");
            System.out.println("Number of models per fragment:");
            for (int i = 0; i <= fragNum; i++) {
                System.out.println("\t- Fragment " + i + ": " + dbFragsNumModels[i] + " models");
            }
        }
    }

    // Read an HMM (binary, HMMER 2.0, not byteswap) and return its length in bytes
    private static long readHMM(BufferedInputStream bis, boolean swap) throws Exception {
        long modelBytes = 0;
        int toSkip;

        // Magic number
        try {
            int magic;
            if (swap) {
                magic = Integer.reverseBytes(readInt(bis));
            } else {
                magic = readInt(bis);
            }
            // Check if it's either HMMER 2.0 in binary, or the swapped version
            if (magic != v20Magic) {
                throw new Exception("Error: unsupported format for binary DB, must be HMMER 2.0");
            }
        } catch (EOFException e) {
            return modelBytes;
        }
        modelBytes += INT_SIZE;
        // Flags
        int flags;
        if (swap) {
            flags = Integer.reverseBytes(readInt(bis));
        } else {
            flags = readInt(bis);
        }
        modelBytes += INT_SIZE;
        // Name
        if (swap) {
            toSkip = Integer.reverseBytes(readInt(bis)) * CHAR_SIZE;
        } else {
            toSkip = readInt(bis) * CHAR_SIZE;
        }
        skip(bis, toSkip);
        modelBytes += (toSkip + INT_SIZE);
        // Accession
        if ((flags & PLAN7_ACC) > 0) {
            if (swap) {
                toSkip = Integer.reverseBytes(readInt(bis)) * CHAR_SIZE;
            } else {
                toSkip = readInt(bis) * CHAR_SIZE;
            }
            skip(bis, toSkip);
            modelBytes += (toSkip + INT_SIZE);
        }
        // Description
        if ((flags & PLAN7_DESC) > 0) {
            if (swap) {
                toSkip = Integer.reverseBytes(readInt(bis)) * CHAR_SIZE;
            } else {
                toSkip = readInt(bis) * CHAR_SIZE;
            }
            skip(bis, toSkip);
            modelBytes += (toSkip + INT_SIZE);
        }
        // Model length
        int modelLength;
        if (swap) {
            modelLength = Integer.reverseBytes(readInt(bis));
        } else {
            modelLength = readInt(bis);
        }
        modelBytes += INT_SIZE;
        // Alphabet type
        readInt(bis);
        modelBytes += INT_SIZE;
        // RF alignment annotation
        if ((flags & PLAN7_RF) > 0) {
            toSkip = (modelLength + 1) * CHAR_SIZE;
            skip(bis, toSkip);
            modelBytes += toSkip;
        }
        // CS alignment annotation
        if ((flags & PLAN7_CS) > 0) {
            toSkip = (modelLength + 1) * CHAR_SIZE;
            skip(bis, toSkip);
            modelBytes += toSkip;
        }
        // Alignment map annotation
        if ((flags & PLAN7_MAP) > 0) {
            toSkip = (modelLength + 1) * INT_SIZE;
            skip(bis, toSkip);
            modelBytes += toSkip;
        }
        // Command line log
        if (swap) {
            toSkip = Integer.reverseBytes(readInt(bis)) * CHAR_SIZE;
        } else {
            toSkip = readInt(bis) * CHAR_SIZE;
        }
        skip(bis, toSkip);
        modelBytes += (toSkip + INT_SIZE);
        // Nseq
        readInt(bis);
        modelBytes += INT_SIZE;
        // Creation time
        if (swap) {
            toSkip = Integer.reverseBytes(readInt(bis)) * CHAR_SIZE;
        } else {
            toSkip = readInt(bis) * CHAR_SIZE;
        }
        skip(bis, toSkip);
        modelBytes += (toSkip + INT_SIZE);
        // Checksum
        readInt(bis);
        modelBytes += INT_SIZE;
        // Pfam gathering thresholds
        if ((flags & PLAN7_GA) > 0) {
            toSkip = 2 * FLOAT_SIZE;
            skip(bis, toSkip);
            modelBytes += toSkip;
        }
        // Pfam trusted cutoffs
        if ((flags & PLAN7_TC) > 0) {
            toSkip = 2 * FLOAT_SIZE;
            skip(bis, toSkip);
            modelBytes += toSkip;
        }
        // Pfam noise cutoffs
        if ((flags & PLAN7_NC) > 0) {
            toSkip = 2 * FLOAT_SIZE;
            skip(bis, toSkip);
            modelBytes += toSkip;
        }
        // Specials
        toSkip = 4 * 2 * FLOAT_SIZE;
        skip(bis, toSkip);
        modelBytes += toSkip;
        // Null model
        toSkip = FLOAT_SIZE + ALPHABET_SIZE * FLOAT_SIZE;
        skip(bis, toSkip);
        modelBytes += toSkip;
        // EVD stats
        if ((flags & PLAN7_STATS) > 0) {
            toSkip = 2 * FLOAT_SIZE;
            skip(bis, toSkip);
            modelBytes += toSkip;
        }
        // Entry/exit probabilities
        toSkip = (2 * modelLength + 3) * FLOAT_SIZE;
        skip(bis, toSkip);
        modelBytes += toSkip;
        // Main model
        toSkip = modelLength * ALPHABET_SIZE * FLOAT_SIZE
                + (modelLength - 1) * ALPHABET_SIZE * FLOAT_SIZE
                + (modelLength - 1) * TRANS_PROB_SIZE * FLOAT_SIZE;
        modelBytes += toSkip;
        skip(bis, toSkip);

        return modelBytes;
    }

    private static int readInt(BufferedInputStream bis) throws Exception {
        int toRead = INT_SIZE;
        int read = 0, sumRead = 0;
        while (toRead > 0) {
            read = bis.read(intBytes, sumRead, toRead);
            if (read > 0) {
                toRead -= read;
                sumRead += read;
            } else if (read == 0) {
                Thread.sleep(100);
            } else {
                throw new EOFException("End of file");
            }
        }

        return (((intBytes[0] & 0xff) << 24)
                | ((intBytes[1] & 0xff) << 16)
                | ((intBytes[2] & 0xff) << 8)
                | (intBytes[3] & 0xff));
    }

    private static void skip(BufferedInputStream bis, int toSkip) throws Exception {
        int skipped;
        while (toSkip > 0) {
            skipped = (int) bis.skip(toSkip);
            if (skipped == 0) {
                throw new EOFException("Error: unexpected end of file");
            }
            toSkip -= skipped;
        }
    }

    private static void prepareResultFile(String source,
            String dest,
            String seqsName,
            String dbName) throws IOException {

        String headerEnd = "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";

        String header = "hmmpfam - search one or more sequences against HMM database\n"
                + "HMMER 2.4i (December 2006)\n"
                + "Copyright (C) 1992-2006 HHMI Janelia Farm\n"
                + "Freely distributed under the GNU General Public License (GPL)\n"
                + "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -\n"
                + "HMM file:                 " + dbName + "\n"
                + "Sequence file:            " + seqsName + "\n"
                + headerEnd + "\n";

        if (debug) {
            System.out.println("\nPreparing result file: ");
            System.out.println("\t- Source: " + source);
            System.out.println("\t- Destination: " + dest);
        }
        source = source.substring(283);
        int index = source.indexOf(headerEnd);
        source = source.substring(index + headerEnd.length());
        // Open the destination file and write the final header
        FileOutputStream dstFos = new FileOutputStream(dest);
        dstFos.write((header + source).getBytes());
        dstFos.close();

    }


    // Class to parse and store the command line arguments of hmmpfam provided by the user
    public static class CommandLineArgs {

        private StringBuilder commandLineArgs;
        private int aLimit;
        private int threshZ;

        public CommandLineArgs(String[] args, int startingPos) {
            commandLineArgs = new StringBuilder("--cpu 0");
            aLimit = -1;
            threshZ = -1;

            for (int i = startingPos; i < args.length; i++) {
                if (args[i].equals("-n")) {
                    System.out.println("-n option not supported, ignoring");
                } else if (args[i].equals("-A")) {
                    commandLineArgs.append(" ").append(args[i]).append(" ").append(args[++i]);
                    aLimit = Integer.parseInt(args[i]);
                } else if (args[i].equals("-E")) {
                    commandLineArgs.append(" ").append(args[i]).append(" ").append(args[++i]);
                } else if (args[i].equals("-T")) {
                    commandLineArgs.append(" ").append(args[i]).append(" ").append(args[++i]);
                } else if (args[i].equals("-Z")) {
                    commandLineArgs.append(" ").append(args[i]).append(" ").append(args[++i]);
                    threshZ = Integer.parseInt(args[i]);
                } else if (args[i].equals("--acc")) {
                    commandLineArgs.append(" ").append(args[i]);
                } else if (args[i].equals("--compat")) {
                    System.out.println("--compat option not supported, ignoring");
                } else if (args[i].equals("--cpu")) {
                    System.out.println("--cpu: ignoring provided value");
                } else if (args[i].equals("--cut_ga")) {
                    commandLineArgs.append(" ").append(args[i]);
                } else if (args[i].equals("--cut_nc")) {
                    commandLineArgs.append(" ").append(args[i]);
                } else if (args[i].equals("--cut_tc")) {
                    commandLineArgs.append(" ").append(args[i]);
                } else if (args[i].equals("--domE")) {
                    commandLineArgs.append(" ").append(args[i]).append(" ").append(args[++i]);
                } else if (args[i].equals("--domT")) {
                    commandLineArgs.append(" ").append(args[i]).append(" ").append(args[++i]);
                } else if (args[i].equals("--forward")) {
                    commandLineArgs.append(" ").append(args[i]);
                } else if (args[i].equals("--null2")) {
                    commandLineArgs.append(" ").append(args[i]);
                } else if (args[i].equals("--pvm")) {
                    System.out.println("--pvm option not supported, ignoring");
                } else if (args[i].equals("--xnu")) {
                    commandLineArgs.append(" ").append(args[i]);
                } else if (args[i].equals("--informat")) {
                    System.out.println("--informat option not supported, ignoring");
                    i++;
                } else {
                    System.err.println("Error: invalid option " + args[i]);
                    System.exit(1);
                }
            }
        }

        public void addArg(String argName, String argValue) {
            commandLineArgs.append(" ").append(argName).append(" ").append(argValue);
        }

        public String getArgs() {
            return commandLineArgs.toString();
        }

        public int getALimit() {
            return aLimit;
        }

        public int getThreshZ() {
            return threshZ;
        }

        public boolean isADefined() {
            return aLimit >= 0;
        }

        public boolean isZDefined() {
            return threshZ >= 0;
        }
    }
}
