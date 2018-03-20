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

public class Shift {
    public double[] cag;
    public int shift;

    public Shift(double[] cag) {
        this.cag = cag;
        this.shift = 0;
    }

    // SHIFT MANAGEMENT
    public void setShift(int a) {
        shift = a;
    }
    public int getShift() {
        return shift;
    }

    // SETTERS
    public void set(int n, double real, double img) {
        cag[2*(n+shift)]= real;
        cag[2*(n+shift)+1]= img;
    }

    // GETTERS
    public double getReal(int n) {
        return  cag[2*(n+shift)];
    }

    public double getImg(int n) {
        return cag[2*(n+shift)+1];
    }


    public void stockham(Shift x, int xd0, double real, double img, int lk, int ny, int i11, int i12, int i21, int i22) {
        double re11, im11, re21, im21;
        int n11, n12, n21, n22;
        for (int k = 0; k < lk; k++) {
            n11 = (i11 + k) * xd0 + x.shift;
            n12 = (i12 + k) * xd0 + x.shift;
            n21 = (i21 + k) * xd0 + shift;
            n22 = (i22 + k) * xd0 + shift;
            for (int j = 0; j < ny; j++) {
                re11 = x.cag[(n11 + j)*2];
                im11 = x.cag[2*(n11 + j)+1];
                re21 = x.cag[2*(n12 + j)];
                im21 = x.cag[2*(n12 + j)+1];
                cag[2*(n21 + j)] = re11 + re21;
                cag[2*(n21 + j)+1] = im11 + im21;
                re11 -= re21;
                im11 -= im21;
                cag[2*(n22 + j)] = re11 * real - im11 * img;
                cag[2*(n22 + j)+1] = re11 * img + im11 * real;
            }
        }
    }
}
