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
package npb.nascg;

import java.text.DecimalFormat;
import java.util.Arrays;

public class NASCG {
    CGProblemClass CGProblem;
    long time;
    Random[] rng;
    double[] tran;
    private double amult = 1220703125.0d;

    double[][] a;
    int[][] colidx;
    int[][] rowstr;
    double[][] x;

    final int cgitmax = 25;
    int l2npcols;
    int[][] reduce_exch_proc;
    int[][] reduce_send_starts;
    int[][] reduce_send_lengths;
    int[][] reduce_recv_starts;
    int[][] reduce_recv_lengths;

    int[] send_start, send_len;
    int[] exch_proc, exch_recv_length;

    double[][] z;
    double[][] p;
    double[][] q;
    double[][] r;
    double[][] w;
    double[][] norm_temp1;
    double[][] norm_temp10;
    int npcols, nprows, nzz;

    public static void main(String[] args)
        throws Exception{
        char problemSize = 'S';
        int np = 4;
        for (int i = 0; i < args.length; i++) {
            if (args[i].compareToIgnoreCase("-np") == 0) {
                np = Integer.parseInt(args[++i]);
            } else if (args[i].compareToIgnoreCase("-class") == 0) {
                if (args[++i].length() == 1) {
                    problemSize = args[i].toUpperCase().charAt(0);
                } else {
                    System.err.println("Parse args error: " + args[i - 1] + " must have a one char parameter");
                    System.exit(1);
                }
            }
        }
        new NASCG(problemSize, np);
    }


    double[] zeta;
    int[] numberOfColumns, numberOfRows;
    int[] firstcol, lastcol, firstrow,lastrow;
    double[][] rnorm;


    public NASCG(char problemSize, int numberProcs){

        CGProblem = ClassFactory.getCGNASClass(problemSize, numberProcs);

        // Check if number of procs is a power of two.
        if (CGProblem.numProcs != 1 &&
            ((CGProblem.numProcs & (CGProblem.numProcs - 1)) != 0)) {
            System.err.println("Error: nbprocs is " + CGProblem.numProcs +
                " which is not a power of two");
            System.exit(1);
        }

        npcols = nprows = NpbMath.ilog2(CGProblem.numProcs) / 2;

        if ((npcols + nprows) != NpbMath.ilog2(CGProblem.numProcs)) {
            npcols += 1;
        }
        npcols = (int) NpbMath.ipow2(npcols);
        nprows = (int) NpbMath.ipow2(nprows);

        // Check npcols parity
        if (npcols != 1 && ((npcols & (npcols - 1)) != 0)) {
            System.err.println("Error: num_proc_cols is " + npcols + " which is not a power of two");
            System.exit(1);
        }

        // Check nprows parity
        if (nprows != 1 && ((nprows & (nprows - 1)) != 0)) {
            System.err.println("Error: num_proc_rows is " + nprows + " which is not a power of two");
            System.exit(1);
        }
        nzz = ((CGProblem.na * (CGProblem.nonzer + 1) * (CGProblem.nonzer + 1)) / CGProblem.numProcs) +
                ((CGProblem.na * (CGProblem.nonzer + 2 + (CGProblem.numProcs / 256))) / npcols);


        rng= new Random[numberProcs];
        tran= new double[numberProcs];
        a= new double[numberProcs][];
        colidx= new int[numberProcs][];
        rowstr= new int[numberProcs][];
        x= new double[numberProcs][];

        reduce_exch_proc= new int[numberProcs][];
        reduce_send_starts= new int[numberProcs][];
        reduce_send_lengths= new int[numberProcs][];
        reduce_recv_starts= new int[numberProcs][];
        reduce_recv_lengths= new int[numberProcs][];

        zeta= new double[CGProblem.iterations+1];

        numberOfColumns= new int[numberProcs];
        numberOfRows= new int[numberProcs];
        firstcol= new int[numberProcs];
        firstrow= new int[numberProcs];
        lastcol= new int[numberProcs];
        lastrow= new int[numberProcs];

        exch_proc= new int[numberProcs];
        exch_recv_length= new int[numberProcs];

        z= new double[numberProcs][];
        p= new double[numberProcs][];
        q= new double[numberProcs][];
        r= new double[numberProcs][];
        w= new double[numberProcs][];
        norm_temp1= new double[numberProcs][];
        norm_temp10= new double[CGProblem.iterations+1][];
        send_start =  new int[numberProcs];
        send_len =  new int[numberProcs];


        rnorm= new double[CGProblem.iterations+1][1];


        for (int rank=0;rank<numberProcs;rank++){
            rng[rank] = new Random();
            tran[rank] = 314159265.0d;

            a[rank] = new double[nzz + 1];
            colidx[rank] = new int[nzz + 1];
            rowstr[rank] = new int[CGProblem.na + 1 + 1];
            x[rank] = new double[(CGProblem.na / nprows) + 2 + 1];
            reduce_exch_proc[rank] = new int[npcols + 1];
            reduce_send_starts[rank] = new int[npcols + 1];
            reduce_send_lengths[rank] = new int[npcols + 1];
            reduce_recv_starts[rank] = new int[npcols + 1];
            reduce_recv_lengths[rank] = new int[npcols + 1];
            z[rank] = new double[(CGProblem.na / nprows) + 2 + 1];
            p[rank] = new double[(CGProblem.na / nprows) + 2 + 1];
            q[rank] = new double[(CGProblem.na / nprows) + 2 + 1];
            r[rank] = new double[(CGProblem.na / nprows) + 2 + 1];
            w[rank] = new double[(CGProblem.na / nprows) + 2 + 1];
            norm_temp1[rank] = new double[2 + 1];

            setup_submatrix_info(rank);
        }


        for (int rank=0;rank<numberProcs;rank++){
            rng[rank].setSeed(tran[rank]);
            rng[rank].setGmult(amult);
            zeta[1] = rng[rank].randlc(); // tran, amult);
            tran[rank] = rng[rank].getSeed();
            makea(CGProblem.na, nzz, a[rank], colidx[rank],rowstr[rank], CGProblem.nonzer, CGProblem.rcond, CGProblem.shift, rank);
            int k;
            int temp;
            for (int i = 1; i <= numberOfRows[rank]; i++) {
                temp = (rowstr[rank][i + 1] - 1);
                for (k = rowstr[rank][i]; k <= temp; k++) {
                    colidx[rank][k] -= firstcol[rank];
                    colidx[rank][k]++;
                }
            }

            java.util.Arrays.fill(x[rank], 1, ((CGProblem.na / nprows) + 1) + 1, 1.0d);
            zeta[1] = 0.0d;

        }

        printStarted(CGProblem.kernelName, CGProblem.problemClassName, new long[] { CGProblem.na}, CGProblem.niter, numberProcs);

        conjGrad(numberProcs,1);

        for (int rank=0;rank<numberProcs;rank++){
            // End of one untimed iteration
            // set starting vector to (1, 1, .... 1)
            x[rank] = new double[(CGProblem.na / nprows) + 2 + 1];
            zeta= new double[CGProblem.iterations+1];;
            Arrays.fill(x[rank], 1, (CGProblem.na / nprows) + 2, 1.0d);

        }

        System.out.println("   iteration           ||r||                 zeta");
        time=System.currentTimeMillis();

        for (int iterations=1;iterations<=CGProblem.iterations;iterations++){
            conjGrad(numberProcs, iterations);
        }
        for (int iterations=1;iterations<=CGProblem.iterations;iterations++){
            DecimalFormat norm = new DecimalFormat("0.00000000000000E00");
            DecimalFormat norm2 = new DecimalFormat("00.00000000000000E00");
            zeta[iterations]=NASCGImpl.getZeta(CGProblem.shift, norm_temp10[iterations]);
            System.out.println("   " + iterations + " \t\t" + norm.format(Math.sqrt(NASCGImpl.getDouble(rnorm[iterations]))) + " \t" +
                norm2.format(zeta[iterations]));
        }

        time=System.currentTimeMillis()-time;
        printEnd(time/(double)1000, getMflops(time), verify());

    }


    public void conjGrad(int numberProcs, int iteration) {

        double[][] rho = new double[numberProcs][1];
        int i;
        int temp;
        double[][] sums=new double[numberProcs/npcols][1];
        double[][] sums1;

        temp = ((CGProblem.na / nprows) + 1);

        double[][] conj1results=new double[numberProcs][];
        for (int rank=0;rank<numberProcs;rank++){
            conj1results[rank]=NASCGImpl.conjGrad1(temp, numberOfColumns[rank], p[rank], q[rank], r[rank], w[rank], x[rank], z[rank]);
            sums[rank/npcols]=NASCGImpl.doubleAddReduce(sums[rank/npcols],conj1results[rank]);
        }
        // The partition submatrix-vector multiply: use workspace w
        for (int cgit = 1; cgit <= cgitmax; cgit++) {

            for (int rank=0;rank<numberProcs;rank++){
                rho[rank] = sums[rank/npcols];
            }

            for (int rank=0;rank<numberProcs;rank++){
                NASCGImpl.conjGrad2(numberOfRows[rank], rowstr[rank], a[rank], colidx[rank], p[rank], w[rank]);
            }


            // Sum the partition submatrix-vec A.p's across rows Exchange and sum piece of w with
            // procs identified in reduce_exch_proc
            double[][] w1= new double[numberProcs][];


            for (i = l2npcols; i >= 1; i--) {
                for (int rank=0;rank<numberProcs;rank++){
                    int rank2=reduce_exch_proc[rank][i];
                    w1[rank]=NASCGImpl.conjGrad3(reduce_send_starts[rank2][i],reduce_recv_starts[rank][i],send_start[rank],reduce_recv_lengths[rank][i],w[rank],w[rank2],q[rank]);
                }
                for (int rank=0;rank<numberProcs;rank++){
                    w[rank]=w1[rank];
                }
            }

            // Exchange piece of q with transpose processor
            if (l2npcols != 0) {
                for (int rank=0;rank<numberProcs;rank++){
                    NASCGImpl.arrayCopy(w[rank], send_start[rank], q[exch_proc[rank]], 1, exch_recv_length[rank]);
                }
            } else {
                for (int rank=0;rank<numberProcs;rank++){
                    NASCGImpl.arrayCopy(w[rank], 0, q[rank], 0, exch_recv_length[rank]);
                }
            }


            //Clear sums fo reuses ...
            sums= new double[numberProcs/npcols][1];
            sums1= new double[numberProcs/npcols][1];
            for (int rank=0;rank<numberProcs;rank++){
                sums1[rank/npcols]=NASCGImpl.doubleAddReduce(sums1[rank/npcols],NASCGImpl.conjGrad4(numberOfColumns[rank], p[rank], q[rank]));
            }
            for (int rank=0;rank<numberProcs;rank++){
                sums[rank/npcols]=NASCGImpl.doubleAddReduce(sums[rank/npcols],NASCGImpl.conjGrad5(rho[rank], sums1[rank/npcols], numberOfColumns[rank], p[rank], q[rank], r[rank], z[rank]));
            }


            for (int rank=0;rank<numberProcs;rank++){
                NASCGImpl.conjGrad6(sums[rank/npcols], rho[rank],numberOfColumns[rank], p[rank], r[rank] );

            }
        }
        // Compute residual norm explicitly: ||r|| = ||x - A.z||
        // First, form A.z
        // The partition submatrix-vector multiply
        for (int rank=0;rank<numberProcs;rank++){
            w[rank]=NASCGImpl.conjGrad7((CGProblem.na / nprows) + 3, numberOfRows[rank], rowstr[rank], a[rank], z[rank], colidx[rank]);
        }

        double[][]w1 = new double[numberProcs][];
        // Sum the partition submatrix-vec A.z's across rows
        for (i = l2npcols; i >= 1; i--) {
            for (int rank=0;rank<numberProcs;rank++){
                int rank2=reduce_exch_proc[rank][i];
                w1[rank]=NASCGImpl.conjGrad3(reduce_send_starts[rank2][i],reduce_recv_starts[rank][i],send_start[rank],reduce_recv_lengths[rank][i],w[rank],w[rank2],r[rank]);
            }
        }

        if (l2npcols != 0) {
            for (int rank=0;rank<numberProcs;rank++){
                NASCGImpl.arrayCopy(w1[rank], send_start[rank], r[exch_proc[rank]], 1, exch_recv_length[rank]);
            }
        } else {
            for (int rank=0;rank<numberProcs;rank++){
                NASCGImpl.arrayCopy(w1[rank], 0, r[rank], 0, exch_recv_length[rank]);
            }
        }

        //Clear sums fo reuses ...
        sums= new double[numberProcs/npcols][1];
        for (int rank=0;rank<numberProcs;rank++){
            norm_temp1[rank]=NASCGImpl.conjGrad8(numberOfColumns[rank], r[rank], x[rank], z[rank]);
            sums[rank/npcols]=NASCGImpl.doubleAddReduce(sums[rank/npcols], norm_temp1[rank]);
        }

        double[][] norm_temp_reduction= new double[numberProcs/npcols][2];

        for (int rank=0;rank<numberProcs;rank++){
           norm_temp_reduction[rank/npcols]=NASCGImpl.doubleArrayAddReduce(norm_temp_reduction[rank/npcols],norm_temp1[rank]);
        }
        for (int rank=0;rank<numberProcs;rank++){
            NASCGImpl.conjGrad9(norm_temp1[rank], norm_temp_reduction[rank/npcols], numberOfColumns[rank], x[rank], z[rank]);
        }


        rnorm[iteration] = sums[0];
        norm_temp10[iteration]=this.norm_temp1[0];
    }































    private void setup_submatrix_info(int rank) {
        int proc_row;
        int proc_col;
        int col_size;
        int row_size;
        int j;
        int div_factor;

        proc_row = rank / npcols;
        proc_col = rank - (proc_row * npcols);

        // If naa evenly divisible by npcols, then it is evenly divisible by nprows
        if ((CGProblem.na / npcols * npcols) == CGProblem.na) { // .eq.
            col_size = CGProblem.na / npcols;
            firstcol[rank] = (proc_col * col_size) + 1;
            lastcol[rank] = firstcol[rank] - 1 + col_size;
            row_size = CGProblem.na / nprows;
            firstrow[rank] = (proc_row * row_size) + 1;
            lastrow[rank] = firstrow[rank] - 1 + row_size;
            // If naa not evenly divisible by npcols, then first subdivide for nprows and then, if
            // npcols not equal to nprows (local_i.e., not a sq number of procs),
            // get col subdivisions by dividing by 2 each row subdivision.
        } else {
            if (proc_row < (CGProblem.na - (CGProblem.na / nprows * nprows))) { // .lt.
                row_size = (CGProblem.na / nprows) + 1;
                firstrow[rank] = (proc_row * row_size) + 1;
                lastrow[rank] = firstrow[rank] - 1 + row_size;
            } else {
                row_size = CGProblem.na / nprows;
                firstrow[rank] = ((CGProblem.na - (CGProblem.na / nprows * nprows)) * (row_size + 1)) +
                    ((proc_row - (CGProblem.na - (CGProblem.na / nprows * nprows))) * row_size) + 1;
                lastrow[rank] = firstrow[rank] - 1 + row_size;
            }
            if (npcols == nprows) {
                if (proc_col < (CGProblem.na - (CGProblem.na / npcols * npcols))) {
                    col_size = (CGProblem.na / npcols) + 1;
                    firstcol[rank] = (proc_col * col_size) + 1;
                    lastcol[rank] = firstcol[rank] - 1 + col_size;
                } else {
                    col_size = CGProblem.na / npcols;
                    firstcol[rank] = ((CGProblem.na - (CGProblem.na / npcols * npcols)) * (col_size + 1)) +
                        ((proc_col - (CGProblem.na - (CGProblem.na / npcols * npcols))) * col_size) + 1;
                    lastcol[rank] = firstcol[rank] - 1 + col_size;
                }
            } else {
                if ((proc_col / 2) < (CGProblem.na - (CGProblem.na / (npcols / 2) * (npcols / 2)))) {
                    col_size = (CGProblem.na / (npcols / 2)) + 1;
                    firstcol[rank] = ((proc_col / 2) * col_size) + 1;
                    lastcol[rank] = firstcol[rank] - 1 + col_size;
                } else {
                    col_size = CGProblem.na / (npcols / 2);
                    firstcol[rank] = ((CGProblem.na - (CGProblem.na / (npcols / 2) * (npcols / 2))) * (col_size + 1)) +
                        (((proc_col / 2) - (CGProblem.na - (CGProblem.na / (npcols / 2) * (npcols / 2)))) * col_size) + 1;
                    lastcol[rank] = firstcol[rank] - 1 + col_size;
                }

                if ((rank % 2) == 0) {
                    lastcol[rank] = firstcol[rank] - 1 + ((col_size - 1) / 2) + 1;
                } else {
                    firstcol[rank] = firstcol[rank] + ((col_size - 1) / 2) + 1;
                    lastcol[rank] = firstcol[rank] - 1 + (col_size / 2);
                }
            }
        }
        numberOfRows[rank] = lastrow[rank] - firstrow[rank] + 1; // added
        numberOfColumns[rank] = lastcol[rank] - firstcol[rank] + 1; // added

        if (npcols == nprows) {
            send_start[rank] = 1;
            send_len[rank] = numberOfRows[rank]; // lastrow - firstrow + 1;
        } else {
            if ((rank & 1) == 0) { // if ((rank % 2) == 0) {
                send_start[rank] = 1;
                send_len[rank] = ((1 + lastrow[rank]) - firstrow[rank] + 1) / 2;
            } else {
                send_start[rank] = (((1 + lastrow[rank]) - firstrow[rank] + 1) / 2) + 1;
                send_len[rank] = numberOfRows[rank] / 2; // (lastrow - firstrow + 1)
                // / 2;
            }
        }
        // Transpose exchange processor
        if (npcols == nprows) {
            exch_proc[rank] = ((rank % nprows) * nprows) + (rank / nprows);
        } else {
            exch_proc[rank] = (2 * ((((rank / 2) % nprows) * nprows) + (rank / 2 / nprows))) + (rank % 2);
        }

        int l2 = npcols / 2;
        l2npcols = 0;
        while (l2 > 0) {
            l2npcols++;
            l2 /= 2;
        }

        // Set up the reduce phase schedules...
        div_factor = npcols;
        for (int i = 1; i <= l2npcols; i++) {
            j = ((proc_col + (div_factor / 2)) % div_factor) + (proc_col / div_factor * div_factor);
            reduce_exch_proc[rank][i] = (proc_row * npcols) + j;

            div_factor = div_factor / 2;
        }

        for (int i = l2npcols; i >= 1; i--) {
            if (nprows == npcols) {
                reduce_send_starts[rank][i] = send_start[rank];
                reduce_send_lengths[rank][i] = send_len[rank];
                reduce_recv_lengths[rank][i] = lastrow[rank] - firstrow[rank] + 1;
            } else {
                reduce_recv_lengths[rank][i] = send_len[rank];
                if (i == l2npcols) {
                    reduce_send_lengths[rank][i] = (lastrow[rank] - firstrow[rank] + 1) - send_len[rank];
                    if ((rank & 1) == 0) { // if ((rank / 2 * 2) == rank) {
                        reduce_send_starts[rank][i] = send_start[rank] + send_len[rank];
                    } else {
                        reduce_send_starts[rank][i] = 1;
                    }
                } else {
                    reduce_send_lengths[rank][i] = send_len[rank];
                    reduce_send_starts[rank][i] = send_start[rank];
                }
            }
            reduce_recv_starts[rank][i] = send_start[rank];
        }
        exch_recv_length[rank] = numberOfColumns[rank] + 1;
    }


    private void makea(int n, int nz, double[] a, int[] colidx, int[] rowstr, int nonzer, double rcond,double shift, int rank) {
        int i;
        int nnza;
        int ivelt;
        int ivelt1;
        int irow;
        int nzv;
        int jcol;
        int iouter;

        int[] iv = new int[(2 * CGProblem.na) + 1 + 1];
        double[] v = new double[CGProblem.na + 1 + 1];
        int[] acol = new int[nzz + 1];
        int[] arow = new int[nzz + 1];
        double[] aelt = new double[nzz + 1];

        // nonzer is approximately (int(sqrt(nnza /n)));
        double size;
        double ratio;
        double scale;

        size = 1.0d;
        ratio = Math.pow(rcond, (1.0 / (float) n));
        nnza = 0;

        // Initialize colidx(n+1 .. 2n) to zero.
        // Used by sprnvc to mark nonzero positions
        Arrays.fill(colidx, n + 1, (2 * n) + 1, 0);

        for (iouter = 1; iouter <= n; iouter++) {
            nzv = nonzer;
            sprnvc(n, nzv, v, iv, colidx, 0, colidx, n, rank);
            nzv = vecset(n, v, iv, nzv, iouter, .5);

            for (ivelt = 1; ivelt <= nzv; ivelt++) {
                jcol = iv[ivelt];
                if ((jcol >= firstcol[rank]) && (jcol <= lastcol[rank])) {
                    scale = size * v[ivelt];
                    for (ivelt1 = 1; ivelt1 <= nzv; ivelt1++) {
                        irow = iv[ivelt1];
                        if ((irow >= firstrow[rank]) && (irow <= lastrow[rank])) {
                            nnza++;
                            if (nnza > nz) {
                                System.out.println("Space for matrix elements exceeded in makea");
                                System.out.println("nnza, nzmax = " + nnza + ", " + nz);
                                System.out.println(" iouter = " + iouter);
                                System.exit(1);
                            }
                            acol[nnza] = jcol;
                            arow[nnza] = irow;
                            aelt[nnza] = v[ivelt1] * scale;
                        }
                    }
                }
            }
            size = size * ratio;
        }

        // ... add the identity * rcond to the generated matrix to bound
        // the smallest eigenvalue from below by rcond
        for (i = firstrow[rank]; i <= lastrow[rank]; i++) {
            if ((i >= firstcol[rank]) && (i <= lastcol[rank])) {
                iouter = n + i;
                nnza++;
                if (nnza > nz) {
                    System.out.println("Space for matrix elements exceeded in makea");
                    System.out.println("nnza, nzmax = " + nnza + ", " + nz);
                    System.out.println(" iouter = " + iouter);
                    System.exit(1);
                    return;
                }
                acol[nnza] = i;
                arow[nnza] = i;
                aelt[nnza] = rcond - shift;
            }
        }

        // ... make the sparse matrix from list of elements with duplicates
        // (v and iv are used as workspace)
        sparse(a, colidx, rowstr, n, arow, acol, aelt, v, iv, 0, iv, n, nnza, rank);
    }

    private void sparse(double[] a, int[] colidx, int[] rowstr, int n, int[] arow, int[] acol, double[] aelt,
            double[] x, int[] mark, int mark_offst, int[] nzloc, int nzloc_offst, int nnza, int rank) {

        // generate a sparse matrix from a list of
        // [col, row, element] tri
        int ind;
        int j;
        int jajp1;
        int nza;
        int k;
        int nzrow;
        double xi;
        int temp;

        // ...count the number of triples in each row
        for (j = 1; j <= n; j++) {
            rowstr[j] = 0;
            mark[j + mark_offst] = 0;
        }
        rowstr[n + 1] = 0;

        for (nza = 1; nza <= nnza; nza++) {
            j = (arow[nza] - firstrow[rank] + 1) + 1;
            rowstr[j]++;
        }

        rowstr[1] = 1;
        temp = numberOfRows[rank] + 1;
        for (j = 2; j <= temp; j++) {
            rowstr[j] = rowstr[j] + rowstr[j - 1];
        }

        // ... rowstr(j) now is the location of the first nonzero of row j of a
        // ... do a bucket sort of the triples on the row index
        for (nza = 1; nza <= nnza; nza++) {
            j = arow[nza] - firstrow[rank] + 1;
            k = rowstr[j];
            a[k] = aelt[nza];
            colidx[k] = acol[nza];
            rowstr[j] = rowstr[j] + 1;
        }

        // ... rowstr(j) now points to the first element of row j+1
        for (j = numberOfRows[rank]; j >= 1; j--) {
            // j--) {
            rowstr[j + 1] = rowstr[j];
        }
        rowstr[1] = 1;

        // ... generate the actual output rows by adding elements
        nza = 0;
        for (ind = 1; ind <= n; ind++) {
            x[ind] = 0.0d;
            mark[ind + mark_offst] = 0;
        }

        jajp1 = rowstr[1];
        for (j = 1; j <= numberOfRows[rank]; j++) {
            nzrow = 0;

            // ...loop over the jth row of a
            for (k = jajp1; k <= (rowstr[j + 1] - 1); k++) {
                ind = colidx[k];
                x[ind] = x[ind] + a[k];
                if ((mark[ind + mark_offst] == 0) && (x[ind] != 0)) {
                    mark[ind + mark_offst] = 1;
                    nzrow = nzrow + 1;
                    nzloc[nzrow + nzloc_offst] = ind;
                }
            }

            // ... extract the nonzeros of this row
            for (k = 1; k <= nzrow; k++) {
                ind = nzloc[k + nzloc_offst];
                mark[ind + mark_offst] = 0;
                xi = x[ind];
                x[ind] = 0.;
                if (xi != 0.) {
                    nza = nza + 1;
                    a[nza] = xi;
                    colidx[nza] = ind;
                }
            }

            jajp1 = rowstr[j + 1];
            rowstr[j + 1] = nza + rowstr[1];
        }
    }

    private int vecset(int n, double[] v, int[] iv, int nzv, int i, double val) {
        boolean set = false;
        for (int k = 1; k <= nzv; k++) {
            if (iv[k] == i) {
                v[k] = val;
                set = true;
            }
        }
        if (!set) {
            nzv = nzv + 1;
            v[nzv] = val;
            iv[nzv] = i;
        }
        return nzv;
    }

    private void sprnvc(int n, int nz, double[] v, int[] iv, int[] nzloc, int nzloc_offst, int[] mark,
            int mark_offst, int rank) {
        int nzrow = 0;
        int nzv = 0;
        int i;
        int ii;
        int nn1 = 1;
        double vecelt;
        double vecloc;

        while (true) {
            nn1 <<= 1; // nn1 = 2 * nn1;
            if (nn1 >= n) {
                break;
            }
        }

        // nn1 is the smallest power of two not less than n
        while (true) {
            if (nzv >= nz) {
                for (ii = 1; ii <= nzrow; ii++) {
                    i = nzloc[ii + nzloc_offst];
                    mark[i + mark_offst] = 0;
                }
                return;
            }

            rng[rank].setSeed(tran[rank]);
            rng[rank].setGmult(amult);
            vecelt = rng[rank].randlc();

            // generate an integer between 1 and n in a portable manner
            vecloc = rng[rank].randlc(); // tran, amult);
            tran[rank] = rng[rank].getSeed();
            i = (int) (nn1 * vecloc) + 1;
            if (i > n) {
                continue;
            }

            if (mark[i + mark_offst] == 0) {
                mark[i + mark_offst] = 1;
                nzrow = nzrow + 1;
                nzloc[nzrow + nzloc_offst] = i;
                nzv = nzv + 1;
                v[nzv] = vecelt;
                iv[nzv] = i;
            }
        }
    }








    public static void printStarted(String kernel, char className, long[] size, int nbIteration, int nbProcess) {
        System.out.print("\n\n NAS Parallel Benchmarks ProActive -- " + kernel + " Benchmark\n\n");
        System.out.println(" Class: " + className);
        System.out.print(" Size:  " + size[0]);
        for (int i = 1; i < size.length; i++) {
            System.out.print(" x " + size[i]);
        }
        System.out.println();

        System.out.println(" Iterations:   " + nbIteration);
        System.out.println(" Number of processes:     " + nbProcess);
    }

    public void printEnd(double totalTime, double mops, boolean passed_verification) {
        String verif;
        String javaVersion = System.getProperty("java.vm.vendor") + " " + System.getProperty("java.vm.name")
                + " " + System.getProperty("java.vm.version") + " - Version " + System.getProperty("java.version");

        verif = passed_verification ? "SUCCESSFUL" : "UNSUCCESSFUL";

        System.out.println("\n\n " + CGProblem.kernelName + " Benchmark Completed");
        System.out.println(" Class            =  " + CGProblem.problemClassName);
        System.out.println(" Size             =  " + CGProblem.size);
        System.out.println(" Iterations       =  " + CGProblem.iterations);
        System.out.println(" Time in seconds  =  " + totalTime);
        System.out.println(" Total processes  =  " + CGProblem.numProcs);
        System.out.println(" Mop/s total      =  " + mops);
        System.out.println(" Mop/s/process    =  " + (mops / CGProblem.numProcs));
        System.out.println(" Operation type   =  " + CGProblem.operationType);
        System.out.println(" Verification     =  " + verif);
        System.out.println(" NPB Version      =  " + CGProblem.version);
        System.out.println(" Java RE          =  " + javaVersion);
    }

    private double getMflops(long total) {
        double ntf = CGProblem.size;
        double time = total;
        double mflops = ntf *
            (14.8157 + 7.19641 * Math.log(ntf) + (5.23518 + 7.21113 * Math.log(ntf)) * CGProblem.niter) / time /
            1000.0;
        return mflops;
    }



        private boolean verify() {
        boolean verified;
        double epsilon = 0.0000000001d;

        if (CGProblem.problemClassName != 'U') {
            if (Math.abs(zeta[CGProblem.iterations] - CGProblem.zeta_verify_value) <= epsilon) {
                verified = true;
                System.out.println(" VERIFICATION SUCCESSFUL");
                System.out.println(" Zeta is    " + zeta[CGProblem.iterations]);
                System.out.println(" Error is   " + (zeta[CGProblem.iterations] - CGProblem.zeta_verify_value));
            } else {
                verified = false;
                System.out.println(" VERIFICATION FAILED");
                System.out.println(" Zeta                " + zeta[CGProblem.iterations]);
                System.out.println(" The correct zeta is " + CGProblem.zeta_verify_value);
            }
        } else {
            verified = false;
            System.out.println(" Problem size unknown");
            System.out.println(" NO VERIFICATION PERFORMED");
        }
        return verified;
    }


}
