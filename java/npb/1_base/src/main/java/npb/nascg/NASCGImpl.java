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


public class NASCGImpl {

    public static double[] conjGrad1(int temp, int numberOfColumns, double[] p, double[] q, double[] r, double[] w, double[] x, double[] z ){
        double[] sum = new double[1];
        int i;
        // Initialize the CG algorithm:

        for (i = 1; i <= temp; i++) {
            q[i] = 0.0d;
            z[i] = 0.0d;
            r[i] = x[i];
            p[i] = r[i];
            w[i] = 0.0d;
        }
        i=temp+1;
        // rho = r.r
        // Now, obtain the norm of r: First, sum squares of r elements locally...


        sum[0]=0;
        for (i = 1; i <= numberOfColumns; i++) {
            sum[0] += (r[i] * r[i]);
        }

        return sum;
    }


    public static double[] doubleAddReduce(double[] val1, double[] val2){
        return new double[] {val1[0]+val2[0]};
    }

    public static void conjGrad2(int numberOfRows, int[] rowstr, double[] a, int[] colidx, double[] p, double[] w){
        double sum=0;
        for (int j = 1; j <= numberOfRows; j++) {
            sum = 0.0d;
            int temp = (rowstr[j + 1] - 1);
            for (int  k = rowstr[j]; k <= temp; k++) {
                sum += (a[k] * p[colidx[k]]);
            }
            w[j] = sum;
        }
    }

    public static double[] conjGrad3(
                    int reduce_send_starts2,
                    int reduce_recv_starts1,
                    int send_start, 
                    int reduce_recv_lengths,
                    double[] w, 
                    double[] w2,
                    double[] q
                    ){
        double[] out = new double[w.length];
        System.arraycopy(w, 0,  out, 0, w.length);
        System.arraycopy(w2, reduce_send_starts2,  q, reduce_recv_starts1, reduce_recv_lengths);
        int temp = ((send_start + reduce_recv_lengths) - 1);

        for (int j = send_start; j <= temp; j++) {
            out[j]+= q[j];
        }
        return out;
    }

    public static void arrayCopy(double[] w, int send_start, double[] q, int recv_start, int length ){
         System.arraycopy(w, send_start, q, recv_start, length);
    }

    public static double[] conjGrad4(int numberOfColumns, double[] p, double[] q) {
            // Obtain p.q
            double sum = 0.0d;
            for (int i = 1; i <= numberOfColumns; i++) {
                sum += (p[i] * q[i]);
            }
            return new double[]{sum};
    }

    public static double[] conjGrad5(double[] rho, double[] sums, int numberOfColumns, double[] p, double[] q, double[] r, double[] z) {
        // Obtain alpha = rho / (p.q)
        double alpha= rho[0] / sums[0];

        // Obtain z = z + alpha*p
        // and r = r - alpha*q
        for (int i = 1; i <= numberOfColumns; i++) {
            z[i] += (alpha * p[i]);
            r[i] -= (alpha * q[i]);
        }
        
        // rho = r.r
        // Now, obtain the norm of r: First, sum squares of r elements locally...
        double sum = 0.0d;
        for (int i = 1; i <= numberOfColumns; i++) {
            sum += (r[i] * r[i]);
        }
        return new double[]{sum};

    }

    public static void conjGrad6(double[] rho, double[] rho0, int numberOfColumns, double[] p, double[] r) {
        // Obtain beta:
        double beta = rho[0] / rho0[0];
        // p = r + beta*p
        for (int i = 1; i <= numberOfColumns; i++) {
            p[i] = r[i] + (beta * p[i]);
        }
    }

    public static double[] conjGrad7(int size, int numberOfRows, int[] rowstr, double[] a, double[] z, int[] colidx) {
        double[] w= new double[size];
        for (int i = 1; i <= numberOfRows; i++) {
            double sum = 0.0d;
            int temp  = (rowstr[i + 1] - 1);
            for (int j = rowstr[i]; j <= temp; j++) {
                sum += (a[j] * z[colidx[j]]);
            }
            w[i] = sum;
        }
        return w;
    }

    public static double[] conjGrad8(int numberOfColumns, double[] r, double[] x, double[]z) {
        double norm_temp1[]= new double[3];

        // At this point, r contains A.z
        norm_temp1[0] = 0.0d;
        for (int i = 1; i <= numberOfColumns; i++) {
            double d = x[i] - r[i];
            norm_temp1[0] += (d * d);
        }
        for ( int i = 1; i <= numberOfColumns; i++) {
            norm_temp1[1] += (x[i] * z[i]);
            norm_temp1[2] += (z[i] * z[i]);
        }

        return norm_temp1;
    }

    public static double[] doubleArrayAddReduce(double[] d, double[] d0) {
        double[] retorn= new double[d.length];
        retorn[0]=d[0]+d0[1];
        retorn[1]=d[1]+d0[2];
        return retorn;
    }

    public static void conjGrad9(double[] norm_temp1, double[] norm_temp_reduction, int numberOfColumns, double[] x, double[] z) {
        norm_temp1[1]=norm_temp_reduction[0];
        norm_temp1[2] = 1.0d / Math.sqrt(norm_temp_reduction[1]);
        // Normalize z to obtain x
        for (int i = 1; i <= numberOfColumns; i++) {
            x[i] = norm_temp1[2] * z[i];
        }
    }


    public static double getZeta(int shift, double[] zeta){
        return shift + (1.0d / zeta[1]);

    }

    public static double getDouble(double[] d) {
        return d[0];
    }
}
