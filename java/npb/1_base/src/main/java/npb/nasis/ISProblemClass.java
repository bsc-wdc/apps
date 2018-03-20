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
package npb.nasis;


public class ISProblemClass implements java.io.Serializable{

    private static final long serialVersionUID = 40L;

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


    public int[] testIndexArray;
    public int[] testRankArray;
    public int numKeys;
    public int maxIterations;
    public int testArraySize;
    public int totalKeysLog2;
    public int maxKeyLog2;
    public int numBucketsLog2;
    public long totalKeys;
    public int maxKey;
    public int numBuckets;
    public int buffersSize;
    
    public int[] getTestIndexArray(){
        return this.testIndexArray;
    }    
    public int[] getTestRankArray(){
        return this.testRankArray;
    }
    public int getNumKeys(){
        return this.numKeys;
    }
    public int getMaxIterations(){
        return this.maxIterations;
    }
    public int getTestArraySize(){
        return this.testArraySize;
    }
    public int getTotalKeysLog2(){
        return this.totalKeysLog2;
    }
    public int getMaxKeyLog2(){
        return this.maxKeyLog2;
    }
    public int getNumBucketsLog2(){
        return this.numBucketsLog2;
    }
    public long getTotalKeys(){
        return this.totalKeys;
    }
    public int getMaxKey(){
        return this.maxKey;
    }    
    public int getNumBuckets(){
        return this.numBuckets;
    }
    public int getBuffersSize(){
        return this.buffersSize;
    }    
    
   
    public void setTestIndexArray(int[] testIndexArray){
        this.testIndexArray=testIndexArray;
    }
    public void setTestRankArray(int[] testRankArray){
        this.testRankArray=testRankArray;
    }
    public void setNumKeys(int numKeys){
        this.numKeys=numKeys;
    }
    public void setMaxIterations(int maxIterations){
        this.maxIterations=maxIterations;
    }
    public void setTestArraySize(int testArraySize){
        this.testArraySize=testArraySize;
    }
    public void setTotalKeysLog2(int totalKeysLog2){
        this.totalKeysLog2=totalKeysLog2;
    }
    public void setMaxKeyLog2(int maxKeyLog2){
        this.maxKeyLog2=maxKeyLog2;
    }
    public void setNumBucketsLog2(int numBucketsLog2){
        this.numBucketsLog2=numBucketsLog2;
    }
    public void setTotalKeys(long totalKeys){
        this.totalKeys=totalKeys;
    }
    public void setMaxKey(int maxKey){
        this.maxKey=maxKey;
    }
    public void setNumBuckets(int numBuckets){
        this.numBuckets=numBuckets;
    }
    public void setBuffersSize(int buffersSize){
        this.buffersSize=buffersSize;
    }
}
