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

//import java.io.BufferedOutputStream;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
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

    private static final float FVDW_MAX = 2;
    private static final float FSOLV_MAX = 7;
    private static final float EPS_MAX = 5;

    private static String natom = null;
    private static String partialParams;

    private static boolean debug;
    private static String binDir;
    private static String dataDir;
    private static String structDir;
    private static String tmpDir;

    public static void main(String args[]) throws Exception {

        debug = Boolean.parseBoolean(args[0]);
        final int fvdw_steps = Integer.parseInt(args[1]);
        final int fsolv_steps = Integer.parseInt(args[2]);
        final int eps_steps = Integer.parseInt(args[3]);
        binDir = args[4];
        dataDir = args[5];
        structDir = args[6];
        tmpDir = args[7];
        String scoreDir = args[8];

        if (debug) {
            System.out.println("Parameters: ");
            System.out.println("- Debug Enabled");
            System.out.println("- FVDW values to test: " + fvdw_steps);
            System.out.println("- FSOLV values to test: " + fsolv_steps);
            System.out.println("- EPS values to test: " + eps_steps);
            System.out.println("- Binary directory: " + binDir);
            System.out.println("- Data directory: " + dataDir);
            System.out.println("- Structures directory: " + structDir);
            System.out.println("- Temporary directory: " + tmpDir);
            System.out.println("- Scores directory: " + scoreDir);
            System.out.println();
        }
        String[] structs = new File(structDir).list();
        final int N = structs.length;

        if (debug) {
            System.out.println(N + " structures found. A total number of " + N * fvdw_steps * fsolv_steps * eps_steps
                    + " simulations will be performed.");
        }
        File scores = new File(scoreDir);

        if (!scores.exists()) {
            scores.mkdir();
        }
        File tmp = new File(tmpDir);

        if (!tmp.exists()) {
            tmp.mkdir();
        }

        final float fvdw_step = FVDW_MAX / fvdw_steps;
        final float fsolv_step = FSOLV_MAX / fsolv_steps;
        final float eps_step = EPS_MAX / eps_steps;

        // read some parameters from files
        initParams();

        // BEGIN SETUP SECTION

        // generate receptor and ligand and run setup
        int s = 0;

        for (String struct : structs) {
            String structPath = structDir + "/" + struct;
            String receptor = tmpDir + "/receptor_" + s;
            String ligand = tmpDir + "/ligand_" + s;
            String topology = tmpDir + "/topology_" + s;
            String coordinates = tmpDir + "/coordinates_" + s;
            s++;

            DiscreteImpl.genReceptorLigand(structPath, binDir, receptor, ligand);
            DiscreteImpl.dmdSetup(receptor, ligand, binDir, dataDir, topology, coordinates);
        }

        // END SETUP SECTION

        // BEGIN PARAMETER SWEEP SECTION

        Queue<String> coeffList = new LinkedList<String>();
        String pydock = dataDir + PYDOCK;
        Queue<String> list = new LinkedList<String>();

        // parameter sweep loops
        for (int i = 1; i <= fvdw_steps; i++) {

            double fvdw = i * fvdw_step;

            for (int j = 1; j <= fsolv_steps; j++) {

                double fsolv = j * fsolv_step;

                for (int k = 1; k <= eps_steps; k++) {

                    double eps = k * eps_step;

                    String params = genParamFile(fvdw, fsolv, eps);

                    // run N simulations for each configuration
                    for (int ii = 1; ii <= N; ii++) {
                        String topology = tmpDir + "/topology_" + ii;
                        String coord = tmpDir + "/coordinates_" + ii;
                        String average = tmpDir + "/average_" + UUID.randomUUID();

                        list.add(average);

                        DiscreteImpl.simulate(params, topology, coord, natom, binDir, dataDir, average);
                    }

                    // merge all averages in a single file
                    while (list.size() > 1) {
                        Queue<String> listAux = new LinkedList<String>();

                        while (list.size() > 1) {
                            String av1 = list.poll();
                            String av2 = list.poll();

                            DiscreteImpl.merge(av1, av2);

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

        partialParams = " &INPUT\n  TSNAP=" + tsnap + "\n  NBLOC=" + nbloc + "\n  TEMP=" + temp + "\n  SEED=" + seed;
    }

    private static String genParamFile(double fvdw, double fsolv, double eps) throws IOException {
        String content = partialParams + "\n  FVDW=" + fvdw + "\n  FSOLV=" + fsolv + "\n  EPS=" + eps + "\n &END\n";

        Path path = Paths.get(tmpDir, "/dmdparm_" + UUID.randomUUID());
        Files.write(path, content.getBytes());

        return path.toString();
    }
}
