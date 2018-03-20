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
package npb.nasft;


public class ClassFactory {

    public static FTProblemClass getFTNASClass(char problemSize, int np){
        FTProblemClass cl = new FTProblemClass();

        switch (problemSize) {
            case 'S':
                cl.nx = 64;
                cl.ny = 64;
                cl.nz = 64;
                cl.niter = 6;
                cl.vdata_real = new double[] { 0., 554.6087004964, 554.6385409189,
            554.6148406171, 554.5423607415, 554.4255039624, 554.2683411902 };
                cl.vdata_imag = new double[] { 0., 484.5363331978, 486.5304269511,
            488.3910722336, 490.1273169046, 491.7475857993, 493.2597244941 };
                break;
            case 'W':
                cl.nx = 128;
                cl.ny = 128;
                cl.nz = 128;
                cl.niter = 6;
                cl.vdata_real = new double[] { 0., 567.3612178944, 563.1436885271,
            559.4024089970, 556.0698047020, 553.0898991250, 550.4159734538 };
                cl.vdata_imag = new double[] { 0., 529.3246849175, 528.2149986629,
            527.0996558037, 526.0027904925, 524.9400845633, 523.9212247086 };
                break;
            case 'A':
                cl.nx = 256;
                cl.ny = 256;
                cl.nz = 128;
                cl.niter = 6;
                cl.vdata_real = new double[] { 0., 504.6735008193, 505.9412319734,
            506.9376896287, 507.7892868474, 508.5233095391, 509.1487099959 };
                cl.vdata_imag = new double[] { 0., 511.4047905510, 509.8809666433,
            509.8144042213, 510.1336130759, 510.4914655194, 510.7917842803 };
                break;
            case 'B':
                cl.nx = 512;
                cl.ny = 256;
                cl.nz = 256;
                cl.niter = 20;
                cl.vdata_real = new double[] { 0., 517.7643571579, 515.4521291263,
            514.6409228649, 514.2378756213, 513.9626667737, 513.7423460082, 513.5547056878, 513.3910925466,
            513.2470705390, 513.1197729984, 513.0070319283, 512.9070537032, 512.8182883502, 512.7393733383,
            512.6691062020, 512.6064276004, 512.5504076570, 512.5002331720, 512.4551951846, 512.4146770029 };
                cl.vdata_imag = new double[] { 0., 507.7803458597, 508.8249431599,
            509.6208912659, 510.1023387619, 510.3976610617, 510.5948019802, 510.7404165783, 510.8576573661,
            510.9577278523, 511.0460304483, 511.1252433800, 511.1968077718, 511.2616233064, 511.3203605551,
            511.3735928093, 511.4218460548, 511.4656139760, 511.5053595966, 511.5415130407, 511.5744692211 };
                break;
            case 'C':
                cl.nx = 512;
                cl.ny = 512;
                cl.nz = 512;
                cl.niter = 20;
                cl.vdata_real = new double[] { 0., 519.5078707457, 515.5422171134,
            514.4678022222, 514.0150594328, 513.7550426810, 513.5811056728, 513.4569343165, 513.3651975661,
            513.2955192805, 513.2410471738, 513.1971141679, 513.1605205716, 513.1290734194, 513.1012720314,
            513.0760908195, 513.0528295923, 513.0310107773, 513.0103090133, 512.9905029333, 512.9714421109 };
                cl.vdata_imag = new double[] { 0., 514.9019699238, 512.7578201997,
            512.2251847514, 512.1090289018, 512.1143685824, 512.1496764568, 512.1870921893, 512.2193250322,
            512.2454735794, 512.2663649603, 512.2830879827, 512.2965869718, 512.3075927445, 512.3166486553,
            512.3241541685, 512.3304037599, 512.3356167976, 512.3399592211, 512.3435588985, 512.3465164008 };
                break;
            case 'D':
                cl.nx = 2048;
                cl.ny = 1024;
                cl.nz = 1024;
                cl.niter = 25;
                cl.vdata_real = new double[] { 0., 512.2230065252, 512.0463975765,
            511.9865766760, 511.9518799488, 511.9269088223, 511.9082416858, 511.8943814638, 511.8842385057,
            511.8769435632, 511.8718203448, 511.8683569061, 511.8661708593, 511.8649768950, 511.8645605626,
            511.8647586618, 511.8654451572, 511.8665212451, 511.8679083821, 511.8695433664, 511.8713748264,
            511.8733606701, 511.8754661974, 511.8776626738, 511.8799262314, 511.8822370068 };
                cl.vdata_imag = new double[] { 0., 511.8534037109, 511.7061181082,
            511.7096364601, 511.7373863950, 511.7680347632, 511.7967875532, 511.8225281841, 511.8451629348,
            511.8649119387, 511.8820803844, 511.8969781011, 511.9098918835, 511.9210777066, 511.9307604484,
            511.9391362671, 511.9463757241, 511.9526269238, 511.9580184108, 511.9626617538, 511.9666538138,
            511.9700787219, 511.9730095953, 511.9755100241, 511.9776353561, 511.9794338060 };
        }

        cl.dims = new int[3 + 1][3 + 1];

        if (np == 1) {
            cl.np1 = 1;
            cl.np2 = 1;
            cl.layout_type = 0;
            for (int i = 1; i <= 3; i++) {
                cl.dims[1][i] = cl.nx;
                cl.dims[2][i] = cl.ny;
                cl.dims[3][i] = cl.nz;
            }
        } else if (np <= cl.nz) {
            cl.np1 = 1;
            cl.np2 = np;
            cl.layout_type = 1;
            cl.dims[1][1] = cl.nx;
            cl.dims[2][1] = cl.ny;
            cl.dims[3][1] = cl.nz;

            cl.dims[1][2] = cl.nx;
            cl.dims[2][2] = cl.ny;
            cl.dims[3][2] = cl.nz;

            cl.dims[1][3] = cl.nz;
            cl.dims[2][3] = cl.nx;
            cl.dims[3][3] = cl.ny;
        } else {
            cl.np1 = cl.nz;
            cl.np2 = np / cl.nz;
            cl.layout_type = 2;
            cl.dims[1][1] = cl.nx;
            cl.dims[2][1] = cl.ny;
            cl.dims[3][1] = cl.nz;

            cl.dims[1][2] = cl.ny;
            cl.dims[2][2] = cl.nx;
            cl.dims[3][2] = cl.nz;

            cl.dims[1][3] = cl.nz;
            cl.dims[2][3] = cl.nx;
            cl.dims[3][3] = cl.ny;
        }

        for (int i = 1; i <= 3; i++) {
            cl.dims[2][i] = cl.dims[2][i] / cl.np1;
            cl.dims[3][i] = cl.dims[3][i] / cl.np2;
        }

        int maxdim = cl.nx;
        if (cl.ny > maxdim) {
            maxdim = cl.ny;
        }
        if (cl.nz > maxdim) {
            maxdim = cl.nz;
        }
        cl.maxdim = maxdim;

        //common variables
        cl.kernelName = "FT";
        cl.problemClassName = problemSize;
        cl.operationType = "floating point";
        cl.numProcs = np;
        cl.np = np;
        cl.iterations = cl.niter;
        cl.sizeStr = cl.nx + "x" + cl.ny + "x" + cl.nz;
        cl.ntotal_f = cl.nx * cl.ny * cl.nz;
        cl.ntdivnp = ((cl.nx * cl.ny) / cl.np) * cl.nz;
        cl.size = (long) cl.nx * (long)cl.ny * (long)cl.nz;
        cl.version = "3.2";

        return cl;
    }

}
