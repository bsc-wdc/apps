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
package es.bsc.genedetection.coreelements;

import java.lang.String;


public class GeneWise{

	public static String runGeneWise(String seqFile, String genomeFile, String other_args){
		 java.lang.String return_type = null;
		String[] cmd = new String[3];
		cmd[0] = "/bin/sh";
		 cmd[1] ="-c";
		 cmd[2] ="/optimis_service/wise2.2.0/src/bin/genewise "+seqFile+" "+genomeFile+" "+other_args+" -gff";
	
		 Process execProc = null;
		 ProcessBuilder pb = new ProcessBuilder(cmd);
		 java.util.Map<String, String> env = pb.environment();
		 env.put("WISECONFIGDIR", "/optimis_service/wise2.2.0/wisecfg/");
		 try {
			 int exitValue = 0;
			 for (int i = 0; i < 10; i++) {
				 System.out.println("Attempt " + i + " out of " + 3);
				 execProc = pb.start();
				 execProc.getOutputStream().close();
	
				 java.io.BufferedInputStream stderr_is_bis = new java.io.BufferedInputStream(execProc.getErrorStream());
				 byte[]stderr_is_b = new byte[1024];
				 while (stderr_is_bis.read(stderr_is_b) >= 0);
				 stderr_is_bis.close();
				 execProc.getErrorStream().close();
	
				 java.io.InputStream stdout_is = execProc.getInputStream();
				 StringBuilder return_type_sb = new StringBuilder();
				 java.io.BufferedReader stdout_is_br = new java.io.BufferedReader(new java.io.InputStreamReader(stdout_is));
				 String stdout_is_line;
				 while ((stdout_is_line = stdout_is_br.readLine()) != null)
				 	return_type_sb.append(stdout_is_line);
				 stdout_is_br.close();
				 return_type = return_type_sb.toString();
	
				 exitValue = execProc.waitFor();
				 System.out.println(exitValue);
				 if (exitValue == 0) {
					 break;
				 }
			 }
			 if (exitValue != 0) {
				 throw new Exception("Exit value is " + exitValue);
			 }
		} catch (Exception e) {
			 e.printStackTrace();
			 System.exit(1);
		}
		 return return_type;
	
	}

	


}
