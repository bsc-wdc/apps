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


public class FTProblemClass implements java.io.Serializable{

    private static final long serialVersionUID = 40L;
    
    public static final int sizeDims = 3;
    
    public String kernelName;

    public char problemClassName;
    public int numProcs;
    public int iterations;
    public long size;
    public String sizeStr;
    public String operationType;
    public String version;

    public String toString() {
        return this.getClass().getName() +
                "\nKERNEL_NAME: " + this.kernelName +
                "\nPROBLEM_CLASS_NAME:" + this.problemClassName +
                "\nNUM_PROCS: " + this.numProcs +
                "\nOPERATION_TYPE: " + this.operationType;
    }

    public String getKernelName(){
        return this.kernelName;
    }
    public void setKernelName(String name){
        this.kernelName=name;
    }


    public char getProblemClassName(){
        return this.problemClassName;
    }
    public void setProblemClassName(char name){
        this.problemClassName=name;
    }

    public int getNumProcs(){
        return this.numProcs;
    }
    public void setNumProcs(int number){
        this.numProcs=number;
    }


    public int getIterations(){
        return this.iterations;
    }
    public void setIterations(int number){
        this.iterations=number;
    }


    public long getSize(){
        return this.size;
    }
    public void setSize(long size){
        this.size=size;
    }


    public String getSizeStr(){
        return this.sizeStr;
    }
    public void setSizeStr(String str){
        this.sizeStr=str;
    }


    public String getOperationType(){
        return this.operationType;
    }
    public void setOperationType(String type){
        this.operationType=type;
    }

    public String getVersion(){
        return this.version;
    }
    public void setVersion(String name){
        this.version=name;
    }


    public int np1;
    public int getNp1(){
        return np1;
    }
    public void setNp1(int np1){
        this.np1= np1;
    }

    public int np2;
    public int getNp2(){
        return np2;
    }
    public void setNp2(int np2){
        this.np2= np2;
    }

    public int np;
    public int getNp(){
        return np;
    }
    public void setNp(int np){
        this.np= np;
    }

    // layout
    public int layout_type;
    public int getLayout_type(){
        return layout_type;
    }
    public void setLayout_type(int layout_type){
        this.layout_type= layout_type;
    }

    public int[][] dims; // 3*3
    public int[][] getDims(){
        return dims;
    }
    public void setDims(int[][] dims){
        this.dims= dims;
    }

    public int niter;
    public int getNiter(){
        return niter;
    }
    public void setNiter(int niter){
        this.niter= niter;
    }

    public int nx; // 1
    public int getNx(){
        return nx;
    }
    public void setNx(int nx){
        this.nx= nx;
    }

    public int ny; // 2
    public int getNy(){
        return ny;
    }
    public void setNy(int ny){
        this.ny= ny;
    }

    public int nz; // 3
    public int getNz(){
        return nz;
    }
    public void setNz(int nz){
        this.nz= nz;
    }

    public int maxdim;
    public int getMaxdim(){
        return maxdim;
    }
    public void setMaxdim(int maxdim){
        this.maxdim= maxdim;
    }

    public double[] vdata_real;
    public double[] getVdata_real(){
        return vdata_real;
    }
    public void setVdata_real(double[] vdata_real){
        this.vdata_real= vdata_real;
    }

    public double[] vdata_imag;
        public double[] getVdata_imag(){
        return vdata_imag;
    }
    public void setVdata_imag(double[] vdata_imag){
        this.vdata_imag= vdata_imag;
    }

    public double ntotal_f; // 1. * nx * ny * nz
    public double getNtotal_f(){
        return ntotal_f;
    }
    public void setNtotal_f(double ntotal_f){
        this.ntotal_f= ntotal_f;
    }

    public int ntdivnp;
    public int getNtdivnp(){
        return ntdivnp;
    }
    public void setNtdivnp(int ntdivnp){
        this.ntdivnp= ntdivnp;
    }

}
