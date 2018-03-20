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

import static npb.nasft.FTProblemClass.*;


public class NASFTImpl {

    public static final int fftblock_default = 16;
    public static final int fftblockpad_default = 18;
    public static final int transblock = 32;
    public static final int transblockpad = 34;

    // values for random data generation
    public static final double d2m46 = Math.pow(0.5, 46);
    public static final long i246m1 = (long) Math.pow(2, 46) - 1;
    public static final double seed = 314159265.;
    public static final double a = 1220703125.;
    public static final double pi = 3.141592653589793238;
    public static final double alpha = .000001;


    public static void FFT_init(char problemSize, int numberProcs, double[] twiddle, int[] d, double[] u, double[] u1, int xstart2, int ystart1, int ystart2, int zstart1, int zstart2)
    throws Exception{
    	FTProblemClass FTProblem = ClassFactory.getFTNASClass(problemSize, numberProcs);
        compute_indexmap(FTProblem, twiddle, d, xstart2, ystart2, zstart2);
        compute_initial_conditions(FTProblem, u1, d, ystart1, zstart1);
        fft_init(FTProblem.dims[1][1], u);

    }


    public static void FFT_layout0(int numberProcs, char problemSize, int maxdim,  int[] dims1, double[] u0, double[] u1,  int fftblock, int fftblockpad, double[] u, double[] twiddle, int xstart2, int ystart1, int ystart2, int zstart1, int zstart2)
    throws Exception{
	    FFT_init(problemSize, numberProcs, twiddle, dims1, u, u1, xstart2, ystart1, ystart2, zstart1, zstart2);
        double[] scratch = new double[NASFTImpl.fftblockpad_default * maxdim * 4];
        cffts1(1, dims1, 0, u1, u1, scratch, fftblock,  fftblockpad, u);
        cffts2(1, dims1, u1, u1, scratch, fftblock,  fftblockpad, u);
        cffts3(1, dims1, 2, u1, u0, scratch, fftblock,  fftblockpad, u);
    }

    public static double[] inverseFFT_layout0(int nx, int ny, int nz, int ntotal_f, int maxdim, int[] dims1, double[] u0, double[] u1, double[] u2,  double[] twiddle, int fftblock, int fftblockpad, double[] u, int xstart0, int xend0, int ystart0, int yend0, int zstart0, int zend0)
    throws Exception{
        evolve(u0, u1, twiddle, dims1);
        double[] scratch = new double[NASFTImpl.fftblockpad_default * maxdim * 4];
        cffts3(-1, dims1, 2, u1, u1, scratch, fftblock,  fftblockpad, u);
        cffts2(-1, dims1, u1, u1, scratch, fftblock,  fftblockpad, u);
        cffts1(-1, dims1, 0, u1, u2, scratch, fftblock,  fftblockpad, u);


        int q, r, s, n;
        double real = 0;
        double img = 0;

        for (int j = 1; j <= 1024; j++) {
            q = (j % nx) + 1;
            if ((q >= xstart0) && (q <= xend0)) {
                r = ((3 * j) % ny) + 1;
                if ((r >= ystart0) && (r <= yend0)) {
                    s = ((5 * j) % nz) + 1;
                    if ((s >= zstart0) && (s <= zend0)) {
                        n = (q - xstart0) + (r - ystart0) * dims1[0] + (s - zstart0) * dims1[0] * dims1[1];
                        real += u2[2*n];
                        img += u2[2*n+1];
                    }
                }
            }
        }
        return new double[]{real/ntotal_f, img/ntotal_f};
    }


    public static double[] FFT_layout1_init(int numberProcs, char problemSize, int maxdim, int d1, int d2, int[] dims1, double[] u0, double[] u1, int fftblock, int fftblockpad, double[] u, double[] twiddle, int xstart2, int ystart1, int ystart2, int zstart1, int zstart2)
    throws Exception{
        FFT_init(problemSize, numberProcs, twiddle, dims1, u, u1, xstart2, ystart1, ystart2, zstart1, zstart2);
        double[] scratch = new double[fftblockpad_default * maxdim * 4];
        cffts1(1, dims1, 0, u1, u1, scratch, fftblock,  fftblockpad, u);
        cffts2(1, dims1, u1, u1, scratch, fftblock,  fftblockpad, u);
        transpose2_local(d1, d2, u1, u0);
        return scratch;
    }

    public static double[] inverseFFT_layout1_init(int maxdim, int d1, int d2, int[] dims1, double[] u0, double[] u1,double[] u2,  double[] twiddle, int fftblock, int fftblockpad, double[] u)
    throws Exception{
        evolve(u0, u1, twiddle, dims1);
        double[] scratch = new double[NASFTImpl.fftblockpad_default * maxdim * 4];
        cffts1(-1, dims1, 2, u1, u1, scratch, fftblock,  fftblockpad, u);
        transpose2_local(d1, d2,  u1, u2);
        return scratch;
    }

    public static void FFT_layout1_end(int np2, int d1, int d2 , int[] dims1, double[] u0, double[] u1, double[] scratch,  int fftblock, int fftblockpad, double[] u)
    throws Exception{
            transpose2_finish(np2, d1, d2, u1, u0);
            cffts1(1, dims1, 2, u0, u0, scratch, fftblock,  fftblockpad, u);
    }
    
    public static double[] inverseFFT_layout1_end(int nx, int ny, int nz, int np2, int ntotal_f, int d1, int d2, int[] dims1, double[] u1, double[] u2, double[] scratch,  int fftblock, int fftblockpad, double[] u,  int xstart0, int xend0, int ystart0, int yend0, int zstart0, int zend0)
    throws Exception{
        transpose2_finish(np2, d1, d2, u1, u2);
        cffts2(-1, dims1, u2, u2, scratch, fftblock,  fftblockpad, u);
        cffts1(-1, dims1, 0, u2, u2, scratch, fftblock,  fftblockpad, u);

        int q, r, s, n;
        double real = 0;
        double img = 0;

        for (int j = 1; j <= 1024; j++) {
            q = (j % nx) + 1;
            if ((q >= xstart0) && (q <= xend0)) {
                r = ((3 * j) % ny) + 1;
                if ((r >= ystart0) && (r <= yend0)) {
                    s = ((5 * j) % nz) + 1;
                    if ((s >= zstart0) && (s <= zend0)) {
                        n = (q - xstart0) + (r - ystart0) * dims1[0] + (s - zstart0) * dims1[0] * dims1[1];
                        real += u2[2*n];
                        img += u2[2*n+1];
                    }
                }
            }
        }
        return new double[]{real/ntotal_f, img/ntotal_f};
    }


    public static void evolve(double[] u0, double[] u1, double[] twiddle, int[] d) {


        int pos, res;
        double tw;

        for (int k = 0; k < d[2]; k++) {
            for (int j = 0; j < d[1]; j++) {
                res = 0+d[0]*j+d[0]*d[1]*k; // this and u1 have the same dimension
                for (int i = 0; i < d[0]; i++) {
                    pos = (res + i);
                    tw = twiddle[res + i];
                    u0[2*pos] *= tw;
                    u0[2*pos+1] *= tw;
                    u1[2*pos] = u0[2*pos];
                    u1[2*pos+1] = u0[2*pos+1];
                }
            }
        }
    }


    private static void compute_indexmap(FTProblemClass FTProblem, double[] twiddle, int[] d, int xstart2, int ystart2, int zstart2)
    throws Exception{

        int i, j, k, ii, ii2, jj, ij2, kk;
        double ap;

        int d0 = d[2*sizeDims];
        int d01 = d[2*sizeDims] * d[2*sizeDims + 1];
        ap = -4.d * NASFTImpl.alpha * NASFTImpl.pi * NASFTImpl.pi;
        int nx2 = FTProblem.nx / 2;
        int ny2 = FTProblem.ny / 2;
        int nz2 = FTProblem.nz / 2;
        if (FTProblem.layout_type == 0) { // xyz layout
            int ilen = FTProblem.dims[1][3];
            int jlen = FTProblem.dims[2][3];
            int klen = FTProblem.dims[3][3];

            for (i = 0; i < ilen; i++) {
                ii = (((i + xstart2) - 1 + nx2) % FTProblem.nx) - nx2;
                ii2 = ii * ii;
                for (j = 0; j < jlen; j++) {
                    jj = (((j + ystart2) - 1 + ny2) % FTProblem.ny) - ny2;
                    ij2 = (jj * jj) + ii2;
                    for (k = 0; k < klen; k++) {
                        kk = (((k + zstart2) - 1 + nz2) % FTProblem.nz) - nz2;
                        twiddle[(i + j * d0 + k * d01)] = Math.exp(ap * ((kk * kk) + ij2));
                    }
                }
            }
        } else if (FTProblem.layout_type == 1) { // zxy layout
            int ilen = FTProblem.dims[2][3];
            int jlen = FTProblem.dims[3][3];
            int klen = FTProblem.dims[1][3];

            for (i = 0; i < ilen; i++) {
                ii = (((i + xstart2) - 1 + nx2) % FTProblem.nx) - nx2;
                ii2 = ii * ii;
                for (j = 0; j < jlen; j++) {
                    jj = (((j + ystart2) - 1 + ny2) % FTProblem.ny) - ny2;
                    ij2 = (jj * jj) + ii2;
                    for (k = 0; k < klen; k++) {
                        kk = (((k + zstart2) - 1 + nz2) % FTProblem.nz) - nz2;
                        twiddle[(k + i * d0 +  j * d01)] = Math.exp(ap * ((kk * kk) + ij2));
                    }
                }
            }
        } else if (FTProblem.layout_type == 2) { // zxy layout
            int ilen = FTProblem.dims[2][3];
            int jlen = FTProblem.dims[3][3];
            int klen = FTProblem.dims[1][3];

            for (i = 0; i < ilen; i++) {
                ii = (((i + xstart2) - 1 + nx2) % FTProblem.nx) - nx2;
                ii2 = ii * ii;
                for (j = 0; j < jlen; j++) {
                    jj = (((j + ystart2) - 1 + ny2) % FTProblem.ny) - ny2;
                    ij2 = (jj * jj) + ii2;
                    for (k = 0; k < klen; k++) {
                        kk = ((k + zstart2) - 1 + (nz2 % FTProblem.nz)) - nz2;
                        twiddle[(k + i * d0 + j * d01)] = Math.exp(ap * ((kk * kk) + ij2));
                    }
                }
            }
        } else {
            throw new Exception ("Not a valid Layout");
        }
    } // compute_indexmap()

    private static void fft_init(int n, double[] u) {
        // compute the roots-of-unity array that will be used for subsequent FFTs.
        int ku;
        int ln;
        int m;
        double t;
        double ti;

        // Initialize the U array with sines and cosines in a manner that permits
        // stride one access at each FFT iteration.
        m = ilog2(n);
        u[0] = m;
        ku = 2;
        ln = 1;

        for (int j = 0; j < m; j++) {
            t = NASFTImpl.pi / ln;

            for (int i = 0; i <= ln - 1; i++) {
                ti = i * t;
                u[2*(i + ku)] = Math.cos(ti);
                u[2*(i + ku)+1] = Math.sin(ti);
            }

            ku += ln;
            ln *= 2;
        }
    }

    private static void compute_initial_conditions(FTProblemClass FTProblem, double[] u0, int[] d, int ystart1, int zstart1) {
        int k;
        double x0;
        double start;
        double an;

        start = NASFTImpl.seed;
        an = ipow46(NASFTImpl.a, 2 * FTProblem.nx, ((zstart1 - 1) * FTProblem.ny) + (ystart1 - 1));
        start = randlc(start, an);
        an = ipow46(NASFTImpl.a, 2 * FTProblem.nx, FTProblem.ny);

        for (k = 0; k < FTProblem.dims[3][1]; k++) {
            x0 = start;
            x0 = vranlc(2 * FTProblem.nx * FTProblem.dims[2][1], x0, NASFTImpl.a, u0, k*d[0]* d[1]);

            if (k != FTProblem.dims[3][1]) {
                start = randlc(start, an);
            }
        }
    }


    private static void transpose2_local(int n1, int n2, double[] xin, double[] xout) {
        // If possible, block the transpose for cache memory systems.
        // How much does this help? Example: R8000 Power Challenge (90 MHz)
        // Blocked version decreases time spend in this routine
        // from 14 seconds to 5.2 seconds on 8 nodes class A.
        if ((n1 < NASFTImpl.transblock) || (n2 < NASFTImpl.transblock)) {
            if (n1 >= n2) {
                
                for (int i = 0; i < n2; i++) {
                    for (int j = 0; j < n1; j++) {
                            xout[2*(i+j*n2)]= xin[2*(j+i*n1)];
                            xout[2*(i+j*n2)+1]= xin[2*(j+i*n1)+1];
                    }
                }
            } else {
                for (int i = 0; i < n1; i++) {
                    for (int j = 0; j < n2; j++) {
                            xout[2*(i+j*n2)]= xin[2*(j+i*n1)];
                            xout[2*(i+j*n2)+1]= xin[2*(j+i*n1)+1];
                    }
                }
            }
        } else {
            for (int j = 0; j < n2; j += NASFTImpl.transblock) {
                for (int i = 0; i < n1; i += NASFTImpl.transblock) {
                    int pos, posi, res;
                    double[][] cache = new double[NASFTImpl.transblock + 1][2*(NASFTImpl.transblock + 1)];

                    for (int jj = 0; jj < NASFTImpl.transblock; jj++) {
                        pos = i+n1*(j + jj);
                        System.arraycopy(xin, 2*pos, cache[jj], 0, 2*NASFTImpl.transblock);
                    }

                    for (int ii = 0; ii < NASFTImpl.transblock; ii++) {
                        res = j+n2*(i + ii);
                        for (int jj = 0; jj < NASFTImpl.transblock; jj++) {
                            posi = (res + jj);
                            xout[2*posi] = cache[jj][2*ii];
                            xout[2*posi+1] = cache[jj][2*ii+1];
                        }
                    }
                }
            }
        }
    }


    private static void transpose2_finish(int np2, int n1, int n2, double[] xin, double[] xout) {
        int ioff;
        for (int p = 0; p <= (np2 - 1); p++) {
            ioff = p * n2;
            for (int j = 0; j < (n1 / np2); j++)
                System.arraycopy(xin, 2*(n2*j+n2*n1/np2*p), xout, 2*(ioff+n2*np2*j),2*n2);
        }
    }


private static void cffts1(int is, int[] d, int dIdx, double[] x, double[] xout, double[] y, int fftblock, int fftblockpad, double[] u)
        throws Exception{
        int posIdx = dIdx * sizeDims;
        int logd1 = ilog2(d[posIdx]);
        int d2mfftblock = d[posIdx + 1] - fftblock;



        for (int k = 0; k < d[posIdx + 2]; k++) {
            for (int jj = 0; jj <= d2mfftblock; jj += fftblock) {
                int n, res,m;
                for (int i = 0; i < fftblock; i++) {
                    res = d[posIdx]*(i + jj)+d[posIdx]*d[posIdx + 1]*k;
                    for (int j = 0; j < d[posIdx]; j++) {
                        n = (res + j);
                        y[2*(i+j*fftblockpad)] = x[2*n];
                        y[2*(i+j*fftblockpad)+1] = x[2*n+1];
                    }
                }
                cfftz(is, logd1, d[posIdx], y, fftblockpad*d[posIdx], fftblock, fftblockpad, u);
                for (int i = 0; i < fftblock; i++) {
                    res = d[posIdx]*(i + jj)+d[posIdx]*d[posIdx + 1]*k;
                    for (int j = 0; j < d[posIdx]; j++) {
                        n =i+ fftblockpad*j;
                        m = (res + j) ;
                        xout[2*m] = y[2*n];
                        xout[2*m+1] = y[2*n+1];
                    }
                }
            }
        }
    }



    private static void cffts2(int is, int[] d,  double[] x, double[] xout, double[] y, int fftblock, int fftblockpad, double[] u)
        throws Exception{
        int posIdx =  sizeDims;
        int logd2 = ilog2(d[posIdx + 1]);

        for (int k = 0; k < d[posIdx + 2]; k++) {
            for (int ii = 0; ii <= (d[posIdx] - fftblock); ii += fftblock) {
                int n, m, res1, res2;
                for (int i = 0; i < d[posIdx + 1]; i++) {
                    res1 = ii+d[posIdx]*i+d[posIdx]*d[posIdx + 1]*k;
                    res2 = i*fftblockpad;
                    for (int j = 0; j < fftblock; j++) {
                        n = (res1 + j);
                        m = res2 + j;
                        y[2*m] = x[2*n];
                        y[2*m+1] = x[2*n+1];
                    }

                }


                cfftz(is, logd2, d[posIdx + 1], y, fftblockpad*d[posIdx + 1], fftblock, fftblockpad, u);

                for (int i = 0; i < d[posIdx + 1]; i++) {
                    res1 = i*fftblockpad;
                    res2 = ii+ i*d[posIdx]+k*d[posIdx]*d[posIdx + 1];
                    for (int j = 0; j < fftblock; j++) {
                        n = res1 + j;
                        m = (res2 + j) ;
                        xout[2*m] = y[2*n];
                        xout[2*m+1] = y[2*n+1];
                    }
                }
            }
        }
    }
    

    private static void cffts3(int is, int[] d, int dIdx, double[] x, double[] xout, double[] y, int fftblock, int fftblockpad, double[] u)
        throws Exception{
        int posIdx = dIdx * sizeDims;
        int logd3 = ilog2(d[posIdx + 2]);

        for (int j = 0; j < d[posIdx + 1]; j++) {
            for (int ii = 0; ii <= (d[posIdx + 1] - fftblock); ii += fftblock) {
                int n, m, res1, res2;
                for (int i = 0; i < d[posIdx + 2]; i++) {

                    res1 = ii+d[posIdx]*j+d[posIdx]*d[posIdx + 1]*i;

                    res2 = i*fftblockpad;

                    for (int res = 0; res < fftblock; res++) {
                        n = (res1 + res);
                        m = res2 + res;
                        y[2*m] = x[2*n];
                        y[2*m+1] = x[2*n+1];
                    }
                }
                cfftz(is, logd3, d[posIdx + 2], y, fftblockpad*d[posIdx + 2], fftblock, fftblockpad, u);
                for (int i = 0; i < d[posIdx + 2]; i++) {
                    res1 =i*fftblockpad;
                    res2 = ii+ j*d[posIdx]+i*d[posIdx]*d[posIdx + 1];
                    for (int res = 0; res < fftblock; res++) {
                        n = res1 + res;
                        m = (res2 + res);
                        xout[2*m] = y[2*n];
                        xout[2*m+1] = y[2*n+1];
                    }
                }


            }
        }

    }


        private static void cfftz(int is, int m, int n, double[] cag, int yshift, int fftblock, int fftblockpad, double[] u)
            throws Exception{
        int mx;


        mx = (int) u[0];
        if (((is != 1) && (is != -1)) || (m < 1) || (m > mx)) {
            throw new Exception("CFFTZ: Either U has not been initialized, or else " +
                "one of the input parameters is invalid" + is + " " + m + " " + mx);
        }

        for (int l = 1; l <= m; l += 2) {
            fftz2(is, l, m, n, fftblock, fftblockpad, u, cag, 0, yshift);

            if (l == m) {
                for (int j = 0; j < n; j++)
                    System.arraycopy(cag, 2*(j*fftblockpad+yshift), cag, 2*(j*fftblockpad), 2*fftblock);
                return;
            }
            fftz2(is, l + 1, m, n, fftblock, fftblockpad, u, cag, yshift, 0);
        }
    }

    private static void fftz2(int is, int l, int m, int n, int ny, int ny1, double[] u, double[] cag, int xshift, int yshift) {
        int n1;
        int lk;
        int li;
        int lj;
        int ku;
        int i11, i12, i21, i22;

        n1 = n / 2;
        lk = (1 << (l - 1));
        li = (1 << (m - l));
        lj = 2 * lk;
        ku = li + 1;

        for (int i = 0; i < li; i++) {
            i11 = (i * lk);
            i12 = i11 + n1;
            i21 = (i * lj);
            i22 = i21 + lk;

            stockham(cag, xshift, yshift, ny1, u[2*(ku + i)], (is < 1) ? -u[2*(ku + i)+1] : u[2*(ku + i)+1], lk, ny, i11, i12, i21, i22);
        }
    }


    public static void stockham(double[] cag, int xshift, int yshift, int xd0, double real, double img, int lk, int ny, int i11, int i12, int i21, int i22) {
        double re11, im11, re21, im21;
        int n11, n12, n21, n22;
        for (int k = 0; k < lk; k++) {
            n11 = (i11 + k) * xd0 + xshift;
            n12 = (i12 + k) * xd0 + xshift;
            n21 = (i21 + k) * xd0 + yshift;
            n22 = (i22 + k) * xd0 + yshift;
            for (int j = 0; j < ny; j++) {
                re11 = cag[(n11 + j)*2];
                im11 = cag[2*(n11 + j)+1];
                re21 = cag[2*(n12 + j)];
                im21 = cag[2*(n12 + j)+1];
                cag[2*(n21 + j)] = re11 + re21;
                cag[2*(n21 + j)+1] = im11 + im21;
                re11 -= re21;
                im11 -= im21;
                cag[2*(n22 + j)] = re11 * real - im11 * img;
                cag[2*(n22 + j)+1] = re11 * img + im11 * real;
            }
        }
    }

    private static final int ilog2(int n) {
        int nn, lg;

        if (n == 1)
            return 0;

        lg = 1;
        nn = 2;
        while (nn < n) {
            nn *= 2;
            lg += 1;
        }
        return lg;
    }


    private static double ipow46(double a, int exp_1, int exp_2) {
        // compute a^exponent mod 2^46
        double r;
        double result;
        double q;
        int n;
        int n2;
        boolean two_pow;

        // Use
        // a^n = a^(n/2)*a^(n/2) if n even else
        // a^n = a*a^(n-1) if n odd
        result = 1;
        if ((exp_2 == 0) || (exp_1 == 0)) {
            return result;
        }
        q = a;
        r = 1;
        n = exp_1;
        two_pow = true;

        while (two_pow) {
            n2 = n / 2;
            if ((n2 * 2) == n) {
                q = randlc(q, q);
                n = n2;
            } else {
                n = n * exp_2;
                two_pow = false;
            }
        }

        while (n > 1) {
            n2 = n / 2;
            if ((n2 * 2) == n) {
                q = randlc(q, q);
                n = n2;
            } else {
                r = randlc(r, q);
                n = n - 1;
            }
        }
        r = randlc(r, q);
        return r;
    }

    private static double vranlc(int n, double x, double a, double[] y, int offset) {
        long Lx = (long) x;
        long La = (long) a;
        for (int i = 0; i < (n / 2); i++) {
            Lx = (Lx * La) & (NASFTImpl.i246m1);
            y[2*(i + offset)]=((double) (NASFTImpl.d2m46 * Lx));
            Lx = (Lx * La) & (NASFTImpl.i246m1);
            y[2*(i + offset)+1]=( (double) (NASFTImpl.d2m46 * Lx));
        }
        return (double) Lx;
    }

    private static double randlc(double x, double a) {
        double r23, r46, t23, t46, t1, t2, t3, t4, a1, a2, x1, x2, z;
        r23 = Math.pow(0.5, 23);
        r46 = r23* r23;
        t23 = 1<< 23;
        t46 = t23*t23;
        // ---------------------------------------------------------------------
        // Break A into two parts such that A = 2^23 * A1 + A2.
        // ---------------------------------------------------------------------
        t1 = r23 * a;
        a1 = (int) t1;
        a2 = a - t23 * a1;
        // ---------------------------------------------------------------------
        // Break X into two parts such that X = 2^23 * X1 + X2, compute
        // Z = A1 * X2 + A2 * X1 (mod 2^23), and then
        // X = 2^23 * Z + A2 * X2 (mod 2^46).
        // ---------------------------------------------------------------------
        t1 = r23 * x;
        x1 = (int) t1;
        x2 = x - t23 * x1;
        t1 = a1 * x2 + a2 * x1;
        t2 = (int) (r23 * t1);
        z = t1 - t23 * t2;
        t3 = t23 * z + a2 * x2;
        t4 = (int) (r46 * t3);
        x = t3 - t46 * t4;
        return x;
    }

    public static void getComplexArray(int i, double[] xin,double[] xin1,double[] xin2,double[] xin3, double[] temp){
    	int size = temp.length/16;
        System.arraycopy(xin, i*size, temp, 0, size);
        System.arraycopy(xin, (i+1)*size, temp, 1*size, size);
        System.arraycopy(xin, (i+2)*size, temp, 2*size, size);
        System.arraycopy(xin, (i+3)*size, temp, 3*size, size);

        System.arraycopy(xin1, i*size, temp, 4*size, size);
        System.arraycopy(xin1, (i+1)*size, temp, 5*size, size);
        System.arraycopy(xin1, (i+2)*size, temp, 6*size, size);
        System.arraycopy(xin1, (i+3)*size, temp, 7*size, size);

        System.arraycopy(xin2, i*size, temp, 8*size, size);
        System.arraycopy(xin2, (i+1)*size, temp, 9*size, size);
        System.arraycopy(xin2, (i+2)*size, temp, 10*size, size);
        System.arraycopy(xin2, (i+3)*size, temp, 11*size, size);

        System.arraycopy(xin3, i*size, temp, 12*size, size);
        System.arraycopy(xin3, (i+1)*size, temp, 13*size, size);
        System.arraycopy(xin3, (i+2)*size, temp, 14*size, size);
        System.arraycopy(xin3, (i+3)*size, temp, 15*size, size);
    }
    public static void setComplexArray(int i, double[] xin, double[] xout, double[] xout1, double[] xout2, double[] xout3) {
    	int size = xin.length/16;
    	System.arraycopy(xin, 0, xout, i*size, size);
    	System.arraycopy(xin, size, xout1, i*size, size);
    	System.arraycopy(xin, 2*size, xout2, i*size, size);
    	System.arraycopy(xin, 3*size, xout3, i*size, size);

    	System.arraycopy(xin, 4*size, xout, (i+1)*size, size);
    	System.arraycopy(xin, 5*size, xout1, (i+1)*size, size);
    	System.arraycopy(xin, 6*size, xout2, (i+1)*size, size);
    	System.arraycopy(xin, 7*size, xout3, (i+1)*size, size);

    	System.arraycopy(xin, 8*size, xout, (i+2)*size, size);
    	System.arraycopy(xin, 9*size, xout1, (i+2)*size, size);
    	System.arraycopy(xin, 10*size, xout2, (i+2)*size, size);
    	System.arraycopy(xin, 11*size, xout3, (i+2)*size, size);

    	System.arraycopy(xin, 12*size, xout, (i+3)*size, size);
    	System.arraycopy(xin, 13*size, xout1, (i+3)*size, size);
    	System.arraycopy(xin, 14*size, xout2, (i+3)*size, size);
    	System.arraycopy(xin, 15*size, xout3, (i+3)*size, size);

    }
    
    public static void reduceComplex(double[] chk1, double[] chk2) {
        chk1[0] += chk2[0];
        chk1[1] += chk2[1];
    }
    
    public static double[] allocComplexArray(int size) {
    	return new double[size];
    }
    
    public static boolean verify(FTProblemClass FTProblem, double[] cplxSums, int i) {
        double err;
        double epsilon = 0.000000000001; // 1.0E-12
        
        System.out.println("T=" + i + "\tChecksum= (" + cplxSums[0]+", "+cplxSums[1]+")\tReference=(" + FTProblem.vdata_real[i]+","+ FTProblem.vdata_imag[i]+")");
        
        err = (cplxSums[0] - FTProblem.vdata_real[i]) / FTProblem.vdata_real[i];
        if (Math.abs(err) > epsilon)
            return false; 

        err = (cplxSums[1] - FTProblem.vdata_imag[i]) / FTProblem.vdata_imag[i];
        if (Math.abs(err) > epsilon)
            return false; 

        return true;
    }
    
    public static int synchronize(double[] d) {
    	return 0;
    }
    
}
