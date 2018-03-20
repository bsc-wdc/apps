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


public class ClassFactory {

    public static CGProblemClass getCGNASClass(char problemSize, int np){
        CGProblemClass cl = new CGProblemClass();

        switch (problemSize) {
            case 'S':
                cl.na = 1400;
                cl.nonzer = 7;
                cl.shift = 10;
                cl.niter = 15;
                cl.zeta_verify_value = 8.5971775078648;
                break;
            case 'W':
                cl.na = 7000;
                cl.nonzer = 8;
                cl.shift = 12;
                cl.niter = 15;
                cl.zeta_verify_value = 10.362595087124;
                break;
            case 'A':
                cl.na = 14000;
                cl.nonzer = 11;
                cl.shift = 20;
                cl.niter = 15;
                cl.zeta_verify_value = 17.130235054029;
                break;
            case 'B':
                cl.na = 75000;
                cl.nonzer = 13;
                cl.shift = 60;
                cl.niter = 75;
                cl.zeta_verify_value = 22.71274548263;
                break;
            case 'C':
                cl.na = 150000;
                cl.nonzer = 15;
                cl.shift = 110;
                cl.niter = 75;
                cl.zeta_verify_value = 28.973605592845;
                break;
            case 'D':
                cl.na = 1500000;
                cl.nonzer = 21;
                cl.shift = 500;
                cl.niter = 100;
                cl.zeta_verify_value = 52.5145321058;
        }

        //common variables
        cl.kernelName = "CG";
        cl.problemClassName = problemSize;
        cl.operationType = "floating point";
        cl.numProcs = np;
        cl.iterations = cl.niter;
        cl.sizeStr = "" + cl.na;
        cl.size = cl.na;

        cl.rcond = 0.1;

        cl.version = "3.2";

        return cl;
    }

}
