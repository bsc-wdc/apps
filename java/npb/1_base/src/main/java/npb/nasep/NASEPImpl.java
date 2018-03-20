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
package npb.nasep;


public class NASEPImpl {

    public static final double R46 = Math.pow(0.5, 46);
    public static final long T46m1 = (long) Math.pow(2, 46) - 1;

    public static void reduce(double[] values, double[] part){
            values[0]+=part[0];
            values[1]+=part[1];
            values[2]+=part[2];
     }

    public static double[] generate (int m, int numberProcs, int rank)
    throws Exception{
        double sx;
        double sy;
        double[] q;
        double gc;
        int nq;
        double[] x;


        Random rng = new Random();

        double an, a, s;
        double t1, t2, t3, t4;
        double x1, x2;
        double[] dum = new double[3 + 1];
        int mk, mm, nn, nk, nk2, np;
        int i, ik, kk, k;
        int no_large_nodes;
        int np_add;
        int k_offset;

        mk = 16;
        mm = m - mk;
        nn = Random.ipow2(mm);
        nk = Random.ipow2(mk);
        nk2 = nk * 2;
        nq = 10;
        a = 1220703125.;
        s = 271828183.;

        x = new double[(nk2) + 1];
        q = new double[nq]; // start to 0

        dum[1] = 1.;
        dum[2] = 1.;
        dum[3] = 1.;



        // Compute the number of "batches" of random number pairs generated
        // per processor. Adjust if the number of processors does not evenly
        // divide the total number
        np = nn / numberProcs;
        no_large_nodes = (nn % numberProcs);
        if (numberProcs < no_large_nodes) {

            np_add = 1;
        } else {
            np_add = 0;
        }
        np = np + np_add;

        if (np == 0) {
            throw new Exception("Too many nodes: " + numberProcs + " " + nn);
        }

        // Call the random number generator functions and initialize
        // the x-array to reduce the effects of paging on the timings.
        // Also, all mathematical functions that are used. Make
        // sure these initializations cannot be eliminated as dead code.
        dum[1] = rng.vranlc(0, dum[1], dum[2], dum, 3 + 1);
        dum[1] = randlc(dum[2], dum[3]);
        java.util.Arrays.fill(x, -1 * Math.pow(10, -99));
        x[0] = 0.0d;

        rng.vranlc1(0, (long) 0, (long) a, x);

        rng.setLSeed((long) a);
        rng.setLGmult((long) a);
        for (int j = 0; j <= mk; j++) {
            rng.lrandlc();
            rng.setLGmult(rng.getLSeed());
        }

        an = (double) rng.getLSeed();
        gc = 0.;
        sx = 0.;
        sy = 0.;

        // Each instance of this loop may be performed independently. We compute
        // the k offsets separately to take into account the fact that some nodes
        // have more numbers to generate than others
        if (np_add == 1) {
            k_offset = (rank * np) - 1;
        } else {
            k_offset = ((no_large_nodes * (np + 1)) + ((rank - no_large_nodes) * np)) - 1;
        }

        for (k = 1; k <= np; k++) {
            kk = k_offset + k;
            t1 = s;
            t2 = an;

            // Find starting seed t1 for this kk.
            for (i = 1; i <= 100; i++) {
                ik = kk / 2;
                if ((2 * ik) != kk) {
                    t1 = (double) (((long) t1 * (long) t2) & NASEPImpl.T46m1);
                }
                if (ik == 0) {
                    break;
                }
                t2 = (double) (((long) t2 * (long) t2) & NASEPImpl.T46m1);
                kk = ik;
            }

            // Compute uniform pseudorandom numbers.

            rng.vranlc1(nk2, (long) t1, (long) a, x);



            // Compute Gaussian deviates by acceptance-rejection method and
            // totally counts in concentric square annuli. This loop is not
            // vectorizable.

            for (i = 1; i <= nk; i++) {
                x1 = 2 * x[2 * i - 1] - 1.;
                x2 = 2 * x[2 * i] - 1.;
                t1 = x1 * x1 + x2 * x2;
                if (t1 <= 1.) {
                    t2 = Math.sqrt(-2 * Math.log(t1) / t1);
                    t3 = (x1 * t2);
                    t4 = (x2 * t2);
                    q[(byte) Math.max(Math.abs(t3), Math.abs(t4))]++;// += 1.;
                    sx += t3;
                    sy += t4;
                }
            }

        }
        for (i=0;i<nq;i++){
            gc+=q[i];
        }
        double[] retorn={sx, sy, gc};
        return retorn;
    }



    private static double randlc(double x, double a) {
        // This routine returns a uniform pseudorandom double precision number in the
        // range (0, 1) by using the linear congruential generator
        //
        // x_{k+1} = a x_k (mod 2^46)
        //
        // where 0 < x_k < 2^46 and 0 < a < 2^46. This scheme generates 2^44 numbers before
        // repeating. The argument A is the same as 'a' in the above formula, and X is the
        // same as x_0. A and X must be odd double precision integers in the range (1, 2^46).
        // The returned value RANDLC is normalized to be between 0 and 1, i.e.
        // RANDLC = 2^(-46) * x_1
        // X is updated to contain the new seed x_1, so that subsequent calls to RANDLC using
        // the same arguments will generate a continuous sequence.
        long Lx;
        long La;

        Lx = (long) x;
        La = (long) a;

        Lx = (Lx * La) & NASEPImpl.T46m1;
        double randlc = NASEPImpl.R46 * Lx;
        // randlcPtr = Lx;

        return randlc;
    }

}
