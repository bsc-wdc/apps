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

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;


public interface NASCGItf {

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    double[] conjGrad1(
        @Parameter
        int temp,
        @Parameter
        int numberOfColumns,
        @Parameter(direction = Direction.INOUT)
        double[] p,
        @Parameter(direction = Direction.INOUT)
        double[] q,
        @Parameter(direction = Direction.INOUT)
        double[] r,
        @Parameter(direction = Direction.INOUT)
        double[] w,
        @Parameter
        double[] x,
        @Parameter(direction = Direction.INOUT)
        double[] z
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    void conjGrad2(
        @Parameter
        int numberOfRows,
        @Parameter
        int[] rowstr,
        @Parameter
        double[] a,
        @Parameter
        int[] colidx,
        @Parameter
        double[] p,
        @Parameter(direction = Direction.INOUT)
        double[] w
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    double[] conjGrad3(
        @Parameter
        int reduce_send_starts2,
        @Parameter
        int reduce_recv_starts1,
        @Parameter
        int send_start,
        @Parameter
        int reduce_recv_lengths,
        @Parameter
        double[] w,
        @Parameter
        double[] w2,
        @Parameter(direction = Direction.INOUT)
        double[] q
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    double[] conjGrad4(
        @Parameter
        int numberOfColumns,
        @Parameter
        double[] p,
        @Parameter
        double[] q
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    double[] conjGrad5(
        @Parameter
        double[] rho,
        @Parameter
        double[] sums,
        @Parameter
        int numberOfColumns,
        @Parameter
        double[] p,
        @Parameter
        double[] q,
        @Parameter(direction = Direction.INOUT)
        double[] r,
        @Parameter(direction = Direction.INOUT)
        double[] z
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    void conjGrad6(
        @Parameter
        double[] rho,
        @Parameter
        double[] rho0,
        @Parameter
        int numberOfColumns,
        @Parameter(direction = Direction.INOUT)
        double[] p,
        @Parameter
        double[] r
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    double[] conjGrad7(
        @Parameter
        int size,
        @Parameter
        int numberOfRows,
        @Parameter
        int[] rowstr,
        @Parameter
        double[] a,
        @Parameter
        double[] z,
        @Parameter
        int[] colidx
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    double[] conjGrad8(
        @Parameter
        int numberOfColumns,
        @Parameter
        double[] r,
        @Parameter
        double[] x,
        @Parameter
        double[]z
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    void conjGrad9(
        @Parameter(direction = Direction.INOUT)
        double[] norm_temp1,
        @Parameter
        double[] norm_temp_reduction,
        @Parameter
        int numberOfColumns,
        @Parameter(direction = Direction.INOUT)
        double[] x,
        @Parameter
        double[] z
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    double[] doubleAddReduce(
        @Parameter
        double[] val1,
        @Parameter
        double[] val2
    );

    @Method(declaringClass = "npb.nascg.NASCGImpl")
    void arrayCopy(
        @Parameter
        double[] w,
        @Parameter
        int send_start,
        @Parameter(direction = Direction.INOUT)
        double[] q,
        @Parameter
        int recv_start,
        @Parameter
        int length
    );

}
