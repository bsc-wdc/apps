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
package luxrender;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;
import java.util.UUID;

import java.nio.channels.FileChannel;

public class LuxRenderImpl {

    // Debug
	private static final boolean debug = true;
	private static final String THREADS_PER_LUXCONSOLE = "1";

      public static void renderPartition(String inputModelFullPathName,
                                  String partialOutputName,
                                  String luxRenderBinary) throws IOException, InterruptedException, Exception{

        if (debug) {
		  System.out.println("\nRunning LuxConsole with parameters:");
		  System.out.println("\t- Binary Path: " + luxRenderBinary);
		  System.out.println("\t- Input LXS Model file: " + inputModelFullPathName);
		  System.out.println("\t- Output partial Model file: " + partialOutputName);
		  
        }

        String cmd = null;
        
        cmd = luxRenderBinary+"luxconsole "+"-t "+THREADS_PER_LUXCONSOLE+" -o "+partialOutputName+" "+inputModelFullPathName;

        if(debug){
		 System.out.println("\nRenderPartition Cmd -> "+cmd);
		 System.out.println(" ");
		}

		Process renderProc = Runtime.getRuntime().exec(cmd);

        byte[] b = new byte[1024];
		int read;

        // Check the proper ending of the process
		int exitValue = renderProc.waitFor();
		if (exitValue != 0) {
			
			   //Splitting the file model path string using a forward slash as delimiter
	    	   StringTokenizer st = new StringTokenizer(inputModelFullPathName, "/");
	    	   String modelFile = null;
	    	   String modelName = null;
	    		    
	    	   while (st.hasMoreElements()){
	    		 modelFile = st.nextToken();
	    	   }
	    	   modelName = modelFile.substring(0, modelFile.length()-4);
			
			BufferedInputStream bisErr = new BufferedInputStream(renderProc.getErrorStream());
			BufferedOutputStream bosErr = new BufferedOutputStream(new FileOutputStream(modelFile + ".err"));

			while ((read = bisErr.read(b)) >= 0) {
				bosErr.write(b, 0, read);
			}

			bisErr.close();
			bosErr.close();

			throw new Exception("Error executing RenderPartition job, exit value is: " + exitValue);
		}
		else{
			
            String renderOutput = null;
			
			//If process finished succesfully -> result renaming			
			String resName = partialOutputName.substring(0, partialOutputName.length()-3);
			String renamedOut = resName+".IT";
			
		    File resFile = new File(resName+".flm");
            File renamedOutFile = new File(renamedOut);
            resFile.renameTo(renamedOutFile);
             
            if(debug){
              System.out.println("LuxConsole Output:\n ");
              BufferedReader is = new BufferedReader(new InputStreamReader(renderProc.getErrorStream()));
              while ((renderOutput = is.readLine()) != null){           	
                    System.out.println(renderOutput);
              }
            }
		}
   }
   
   
   public static void mergePartitions(String luxRenderBinary, String fA, 
				 String fB, String mergedFile) throws IOException, InterruptedException, Exception
   {   
       String cmd = null;

       //merge of 2 files
       
       //Change per renaming?
  	   File fAFile = new File(fA);
       File fBFile = new File(fB);
       File fAFileRen = new File(fA+".flm");
       File fBFileRen = new File(fB+".flm");      
       //copyFile(fAFile, new File(fA+".flm"));
       //copyFile(fBFile, new File(fB+".flm"));
       fAFile.renameTo(fAFileRen);
       fBFile.renameTo(fBFileRen);
       
       //String mFile = "luxMerge_"+UUID.randomUUID()+".flm";
    		         
       //cmd = luxRenderBinary+"luxmerger "+"-o "+mFile+" "+fA+".flm"+" "+fB+".flm";   		
        cmd = luxRenderBinary+"luxmerger "+"-o "+fA+".flm"+" "+fA+".flm"+" "+fB+".flm";   

       if(debug){
    		System.out.println("\nLuxMerger Cmd -> "+cmd);
    		System.out.println(" ");
       }

       Process mergeProc = Runtime.getRuntime().exec(cmd);

       byte[] b = new byte[1024];
       int read;

       // Check the proper ending of the process
       int exitValue = mergeProc.waitFor();
       if (exitValue != 0) {
    				
    	  //Splitting the file model path string using a forward slash as delimiter
    	  StringTokenizer st = new StringTokenizer(mergedFile, "/");
    	  String outputFile = null;
    	  String outputName = null;
    		    		    
    	  while (st.hasMoreElements()){
    		 outputFile = st.nextToken();
    	  }
    	  outputName = outputFile.substring(0, outputFile.length()-4);
    				
    	  BufferedInputStream bisErr = new BufferedInputStream(mergeProc.getErrorStream());
    	  BufferedOutputStream bosErr = new BufferedOutputStream(new FileOutputStream(outputFile + ".err"));

    	  while ((read = bisErr.read(b)) >= 0) {
    		 bosErr.write(b, 0, read);
    	  }

    	  bisErr.close();
    	  bosErr.close();

    		 throw new Exception("Error executing LuxMerger task, exit value is: " + exitValue);
    	  }
    		 
    	 //If process finished succesfully -> result renaming					
	     //File resFile = new File(mFile);
         File resFile = new File(fA+".flm");
         File renamedOutFile = new File(mergedFile);
         resFile.renameTo(renamedOutFile);
    		 
         //Flushing intermediate files
    	 //File f_a = new File(fA+".flm");
    	 File f_b = new File(fB+".flm");
         //f_a.delete();
         f_b.delete();
        	 		 
         //f_b = new File(fB);
         //f_b.delete();	
   }        

  /*public static void copyFile(File sourceFile, File destFile) throws IOException {

	   FileChannel source = null;
	   FileChannel destination = null;
	   
	   try {
	    source = new FileInputStream(sourceFile).getChannel();
	    destination = new FileOutputStream(destFile).getChannel();
	    destination.transferFrom(source, 0, source.size());
	   }
	   finally {
	     if(source != null) {
	      source.close();
	     }
	     if(destination != null) {
	      destination.close();
	     }
	   }
  }
*/
}
