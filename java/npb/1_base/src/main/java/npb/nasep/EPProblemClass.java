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

/**
 *
 * @author flordan
 */
public class EPProblemClass implements java.io.Serializable{

    private static final long serialVersionUID = 40L;
    
    
    public String kernelName;

    public char problemClassName;
    public int numProcs;
    public int iterations;
    public long size;
    public String sizeStr;
    public String operationType;
    public String version;


    
    public int m;
    public int npm;

    public String toString() {
        return this.getClass().getName() + "\nKERNEL_NAME: " + this.kernelName + "\nPROBLEM_CLASS_NAME:" +
            this.problemClassName + "\nNUM_PROCS: " + this.numProcs + "\nOPERATION_TYPE: " +
            this.operationType;
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

    
    public int getM(){
        return this.m;
    }
    public void setM(int m){
        this.m=m;
    }

    
    public int getNpm(){
        return this.npm;
    }
    public void setNpm(int npm){
        this.npm=npm;
    }

}
