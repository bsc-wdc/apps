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
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Stack;
import java.util.UUID;
import java.util.StringTokenizer;

import luxrender.LuxRenderImpl;

public class LuxRender {

  private static boolean debug;
  private static ArrayList<String> partialOutputs = new ArrayList<String>(); 
  
  public static void main(String args[]) throws Exception {
     	
		/* Parameters:
		 * - 0: Debug
		 * - 1: LuxRender binaries location
		 * - 2: Input LXS full path model
		 * - 3: Number of Worker processes
		 * - 4: Temporary directory
		 * - 5: Output file
		*/
	  
		debug = Boolean.parseBoolean(args[0]);
		String luxRenderBinary = args[1];
		String inputModelFullPathName = args[2];
		int nWorkerProcesses = Integer.parseInt(args[3]);
		String temporaryDir = args[4];
		String outputFullPathName = args[5];
		
		print_header();
		
		if (debug) {
		   System.out.println("Application Parameters: ");
		   System.out.println("- Debug: Enabled");
		   System.out.println("- LuxRender binary path: " + luxRenderBinary);
           System.out.println("- LXS File of Input Model: " + inputModelFullPathName);
           System.out.println("- Temporary Directory: " + temporaryDir);
           System.out.println("- Output File: " + outputFullPathName);
		   System.out.println(" ");
		}
      
       try
       {   	  
    	  //Splitting the file model path string using a forward slash as delimiter
    	   StringTokenizer st = new StringTokenizer(inputModelFullPathName, "/");
    	   String modelFile = null;
    		    
    	   while (st.hasMoreElements()){
    		 modelFile = st.nextToken();
    	   }
    	   
    	   System.out.println("Starting render of "+modelFile+" model:\n");
    	   Long startTime = System.currentTimeMillis();
    		 
    	   for(int i=0;i < nWorkerProcesses; i++){
    	     
    	      //Creating partial output file name  		   
    		   String partialOutputFullPathName = temporaryDir+"luxout_"+UUID.randomUUID();
    		    		   
    		   //Adding extension FLM to output file name
    		   partialOutputFullPathName+=".flm";    		   
    		   partialOutputs.add(partialOutputFullPathName); 	
    		   
    		   //Submitting the job
      		   System.out.println("Rendering task "+(i+1)+" of "+nWorkerProcesses);  
      		   LuxRenderImpl.renderPartition(inputModelFullPathName, partialOutputFullPathName, luxRenderBinary);   	   
    	  } //End for
    	  
    	   if(debug)
           System.out.println("\nMerging partial render tasks: ");
           String lastMerge = "";
           try{
           	   // Final Assembly process -> Merge 2 by 2
                   int neighbor=1;
                   while (neighbor<partialOutputs.size()){
                      for (int result=0; result<partialOutputs.size(); result+=2*neighbor){
                          if (result+neighbor < partialOutputs.size()){
                        	  LuxRenderImpl.mergePartitions(luxRenderBinary, partialOutputs.get(result), partialOutputs.get(result+neighbor),partialOutputs.get(result));
                        	  if(debug)
                        	  System.out.println(" - Merging files -> "+partialOutputs.get(result)+ " and "+partialOutputs.get(result+neighbor));
                        	  lastMerge = partialOutputs.get(result);
                          }
                      }
                      neighbor*=2;
                   }
           	}
           	catch (Exception e){
           	    System.out.println("Error assembling partial results to final result file.");
           	    e.printStackTrace();
           	}

    	   System.out.println("\nWaiting for completion of merge tasks...");
    		   
    	   //Renaming final output name partialMergedFileName -> outputFullPathName	
    	   String outputFileName = outputFullPathName.substring(0, (outputFullPathName.length()-4));
    	   FileInputStream fis = new FileInputStream(lastMerge);
   		   
    	   if(debug)
    	   System.out.println("\nMoving last merged file: "+lastMerge+" to "+outputFileName+".flm");
           
    	   copyFile(fis, new File(outputFileName+".flm"));
          
           //flm to png conversion
           flmToPng(luxRenderBinary, inputModelFullPathName, outputFullPathName);
          
           //Final Cleanup
           cleanUp();
          
           Long stopTime = System.currentTimeMillis();
           Long renderTime = (stopTime - startTime)/1000;
           System.out.println("\n"+modelFile+" model render finished successfully in "+renderTime+" seconds \n");
      }
      catch (Exception e)
      {
          System.out.println("Error on main rendering process");
    	  e.printStackTrace();
      }
  }
  
  
 private static void flmToPng(String luxRenderBinary, String inputModelFullPathName, String outputFullPathName) throws IOException, InterruptedException, Exception{
	 
	    System.out.println("\nCreating FLM to PNG .lxs conversion file");
	 
	 	String inputFileName = inputModelFullPathName.substring(0, (inputModelFullPathName.length()-4));
	 	
	 	String pnglxsOut = inputFileName+"-PNG_"+UUID.randomUUID()+".lxs";
	 	partialOutputs.add(pnglxsOut);
	 	
	 	String outputFileName = outputFullPathName.substring(0, (outputFullPathName.length()-4));
	 
	    File file = new File(inputModelFullPathName);
	    BufferedReader reader = new BufferedReader(new FileReader(file));
	    BufferedWriter bw = new BufferedWriter(new FileWriter(pnglxsOut));
	    String line = "";

	    while((line = reader.readLine()) != null)
	    {   
	       if(line.contains("\"bool write_resume_flm\"")){
	 	    	 line = " \"bool write_resume_flm\" [\"false\"]";
	 	   }
	       
	       if(line.contains("\"bool write_png\"")){
	 	    	 line = " \"bool write_png\" [\"true\"]";
	 	   }  	    	
	    	
	       if(line.contains("\"string filename\"")){
	    	 line = " \"string filename\" [\""+
	    	 		outputFileName+"\"]";
	       }
	       
	       if(line.contains("\"integer halttime\"")){
	 	    	 line = " \"integer halttime\" [1]";
	 	   }
	       
	      bw.write(line);
	      bw.newLine();
	    }
	   reader.close();
	   bw.close();
	
	 System.out.println("Converting FLM result file to a PNG picture on: "+outputFullPathName);	
	 
	 outputFullPathName = outputFileName+".flm";
	 
	 String cmd = luxRenderBinary+"luxconsole -R "+outputFullPathName+" "+pnglxsOut;
	 
	 if(debug){
		 System.out.println("\nFLM to PNG Conversion Cmd -> "+cmd);
		 System.out.println(" ");
	 }

	 Process flmtoPngProc = Runtime.getRuntime().exec(cmd);

     byte[] b = new byte[1024];
		int read;

     // Check the proper ending of the process
	 int exitValue = flmtoPngProc.waitFor();
	 if (exitValue != 0) {
			
			   //Splitting the file model path string using a forward slash as delimiter
	    	   StringTokenizer st = new StringTokenizer(outputFullPathName, "/");
	    	   String outputFile = null;
	    	   String outputName = null;
	    		    
	    	   while (st.hasMoreElements()){
	    		 outputFile = st.nextToken();
	    	   }
	    	   outputName = outputFile.substring(0, outputFile.length()-4);
			
			BufferedInputStream bisErr = new BufferedInputStream(flmtoPngProc.getErrorStream());
			BufferedOutputStream bosErr = new BufferedOutputStream(new FileOutputStream(outputFile + ".err"));

			while ((read = bisErr.read(b)) >= 0) {
				bosErr.write(b, 0, read);
			}

			bisErr.close();
			bosErr.close();

			throw new Exception("Error converting FLM to PNG picture, exit value is: " + exitValue);
		}
 }
 
 private static void cleanUp(){	 
   //Cleaning intermediate files
   System.out.println("Cleaning-up temporary files");  
   for(int i=0; i < partialOutputs.size(); i++){
 		 File f = new File(partialOutputs.get(i));
 		 f.delete();
   } 
 }
 
 private static void copyFile(FileInputStream sourceFile, File destFile) throws IOException {

	   FileChannel source = null;
	   FileChannel destination = null;
	   
	   try {
	    source = sourceFile.getChannel();
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
 
 private static void  print_header(){
  System.out.println("\nCOMPSs LuxRender Rendering Tool:\n");
 } 

}
