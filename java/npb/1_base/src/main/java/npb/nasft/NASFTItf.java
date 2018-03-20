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

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;


public interface NASFTItf {

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    void FFT_layout0(
        @Parameter
        int numberProcs,
        @Parameter
        char problemSize,
        @Parameter
        int maxdim,
        @Parameter
        int[] dims1,
        @Parameter(direction = Direction.INOUT)
        double[] u0,
        @Parameter(direction = Direction.INOUT)
        double[] u1,
        @Parameter
        int fftblock,
        @Parameter
        int fftblockpad,
        @Parameter(direction = Direction.INOUT)
        double[] u,
        @Parameter(direction = Direction.INOUT)
        double[] twiddle,
        @Parameter
        int xstart2,
        @Parameter
        int ystart1,
        @Parameter
        int ystart2,
        @Parameter
        int zstart1,
        @Parameter
        int zstart2
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    double[] inverseFFT_layout0(
        @Parameter
        int nx,
        @Parameter
        int ny,
        @Parameter
        int nz,
        @Parameter
        int ntotal_f,
        @Parameter
        int maxdim,
        @Parameter
        int[] dims1,
        @Parameter(direction = Direction.INOUT)
        double[] u0,
        @Parameter(direction = Direction.INOUT)
        double[] u1,
        @Parameter(direction = Direction.INOUT)
        double[] u2,
        @Parameter
        double[] twiddle,
        @Parameter
        int fftblock,
        @Parameter
        int fftblockpad,
        @Parameter(direction = Direction.INOUT)
        double[] u,
        @Parameter
        int xstart0,
        @Parameter
        int xend0,
        @Parameter
        int ystart0,
        @Parameter
        int yend0,
        @Parameter
        int zstart0,
        @Parameter
        int zend0
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    double[] FFT_layout1_init(
        @Parameter
        int numberProcs,
        @Parameter
        char problemSize,
        @Parameter
        int maxdim,
        @Parameter
        int d1,
        @Parameter
        int d2,
        @Parameter
        int[] dims1,
        @Parameter(direction = Direction.INOUT)
        double[] u0,
        @Parameter(direction = Direction.INOUT)
        double[] u1,
        @Parameter
        int fftblock,
        @Parameter
        int fftblockpad,
        @Parameter(direction = Direction.INOUT)
        double[] u,
        @Parameter(direction = Direction.INOUT)
        double[] twiddle,
        @Parameter
        int xstart2,
        @Parameter
        int ystart1,
        @Parameter
        int ystart2,
        @Parameter
        int zstart1,
        @Parameter
        int zstart2
    );


    @Method(declaringClass = "npb.nasft.NASFTImpl")
    double[] inverseFFT_layout1_init(
        @Parameter
        int maxdim,
        @Parameter
        int d1,
        @Parameter
        int d2,
        @Parameter
        int[] dims1,
        @Parameter(direction = Direction.INOUT)
        double[] u0,
        @Parameter(direction = Direction.INOUT)
        double[] u1,
        @Parameter(direction = Direction.INOUT)
        double[] u2,
        @Parameter
        double[] twiddle,
        @Parameter
        int fftblock,
        @Parameter
        int fftblockpad,
        @Parameter
        double[] u
    );


    @Method(declaringClass = "npb.nasft.NASFTImpl")
    void FFT_layout1_end(
        @Parameter
        int np2,
        @Parameter
        int d1,
        @Parameter
        int d2,
        @Parameter
        int[] dims1,
        @Parameter(direction = Direction.INOUT)
        double[] u0,
        @Parameter(direction = Direction.INOUT)
        double[] u1,
        @Parameter
        double[] scratch,
        @Parameter
        int fftblock,
        @Parameter
        int fftblockpad,
        @Parameter
        double[] u
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    double[] inverseFFT_layout1_end(
        @Parameter
        int nx,
        @Parameter
        int ny,
        @Parameter
        int nz,
        @Parameter
        int np2,
        @Parameter
        int ntotal_f,
        @Parameter
        int d1,
        @Parameter
        int d2,
        @Parameter
        int[]dims1,
        @Parameter(direction = Direction.INOUT)
        double[] u1,
        @Parameter(direction = Direction.INOUT)
        double[] u2,
        @Parameter
        double[] scratch,
        @Parameter
        int fftblock,
        @Parameter
        int fftblockpad,
        @Parameter
        double[] u,
        @Parameter
        int xstart0,
        @Parameter
        int xend0,
        @Parameter
        int ystart0,
        @Parameter
        int yend0,
        @Parameter
        int zstart0,
        @Parameter
        int zend
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    void getComplexArray(
    	@Parameter
        int i,
        @Parameter
        double[] xin,
        @Parameter
        double[] xin1,
        @Parameter
        double[] xin2,
        @Parameter
        double[] xin3,
        @Parameter(direction = Direction.INOUT)
        double[] temp
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    void setComplexArray(
	@Parameter
        int i,
        @Parameter
        double[] xin,
        @Parameter(direction = Direction.INOUT)
        double[] xout,
        @Parameter(direction = Direction.INOUT)
        double[] xout1,
        @Parameter(direction = Direction.INOUT)
        double[] xout2,
        @Parameter(direction = Direction.INOUT)
        double[] xout3
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    void reduceComplex(
        @Parameter(direction = Direction.INOUT)
        double[] chk1,
        @Parameter
        double[] chk2
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    double[] allocComplexArray(
		@Parameter
        int size
    );

    @Method(declaringClass = "npb.nasft.NASFTImpl")
    int synchronize(
		@Parameter
        double[] d
    );

}
