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
package hrt;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;

public class HRTImpl {

    // Debug
	private static final boolean debug = true;

      public static void modeling(String scriptPath, 
    		  					  String confFile,
                                  String user,
                                  int index,
                                  String modelLog) throws IOException, InterruptedException, Exception{
    	  
    	// Create a temp dir
        File tempDir = File.createTempFile("hrt", "tmp");
        if (!tempDir.delete() || !tempDir.mkdir()) {
            throw new IOException("Error creating temp dir");
        }

        if (debug) {
		  System.out.println("\n* Running HRT modelling " + index +" with parameters:");
		  System.out.println("  - HRT Script location: " + scriptPath);
		  System.out.println("  - Configuration file: " + confFile);
		  System.out.println("  - Temporary directory: "+tempDir.getAbsolutePath()+"/");
		  System.out.println("  - User: " + user);
		  System.out.println("  - Model file: " + modelLog);
        }
              
        String cmd = scriptPath+ " -f " + confFile + " -o ./"+tempDir.getName()+" -u "+ user + " -i " + index;

		if(debug){
		 System.out.println("\n* HRT Cmd -> "+cmd);
		 System.out.println(" ");
		}
		
		Long startHrt = System.currentTimeMillis();
		Process hrtProc = Runtime.getRuntime().exec(cmd);
		
        //Check the proper ending of the process
		hrtProc.waitFor();
		
		Long hrtTime = (System.currentTimeMillis()- startHrt)/1000;    
	    
		Process proc = Runtime.getRuntime().exec(cmd);

        proc.waitFor();

        printStream(proc.getInputStream());
        printStream(proc.getErrorStream());
        
        //Renaming output files
        System.out.println(" * Renaming "+tempDir.getAbsolutePath()+"/log_"+index+".log to "+modelLog);
        
        File monitor = new File(tempDir.getAbsolutePath()+"/log_"+index+".log");
        monitor.renameTo(new File(modelLog));
        
        //Removing temporary directory
        tempDir.delete();
        
        System.out.println("\n * Simulation finished in "+hrtTime+" seconds.");
        System.out.println(" ");     
   }
      
   public static void genConfigFile(String startDate, String duration, String confFile) throws IOException, InterruptedException, Exception{

    	  if (debug) {
    		 System.out.println("\n* Running HRT configuration file modification: ");
			 System.out.println("   - Configuration File: " + confFile);
    	  }
    	  
    	  System.out.println(" \n* Writing configuration File with:");
    	  System.out.println("   - Start date: "+startDate);
    	  System.out.println("   - Duration: "+duration);
    	  
    	  FileWriter fstream = new FileWriter(confFile);
    	  BufferedWriter out = new BufferedWriter(fstream);
    	  out.write("# the starting date of the simulation YYYYMMDD\n");
    	  out.write("start_date: "+startDate+"\n");
    	  out.write("# the duration of the simulation YYMMDD number of years, months and days\n");
    	  out.write("duration: "+duration+"\n");
    	  
    	  //Close the output stream
    	  out.close();
    	 
    	  System.out.println(" ");
   }
   
   public static void mergeMonitorLogs(String fileA, String fileB){
	    
	     String line = null;
	     boolean append = true;
	     
	     try
	     {    	 
	         BufferedWriter bw = new BufferedWriter(new FileWriter(fileA, append));
	         BufferedReader bfB = new BufferedReader(new FileReader(fileB));
	         
	         if(debug){
		       System.out.println("\n * Merging monitoring logs -> "+fileB+" to "+fileA);
		     } 	       
	         
	         while ((line = bfB.readLine()) != null)
	         {
	     	    bw.write(line);
	     	    bw.newLine();
	         }
	                  	         
	    	 //Closing final output file
	         bfB.close(); 
	    	 bw.close();
	     }
	     catch (Exception e){
	       System.out.println("Error merging monitoring logs.");
	     }
   }
      
  private static void printStream(InputStream str) throws IOException,Exception {

          BufferedReader stderr = new BufferedReader(new InputStreamReader(str));
          String line;

          while ((line = stderr.readLine()) != null)
                  System.out.println(line);

          stderr.close();
  }
      
  
}
