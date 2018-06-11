package worker.hmmerobj;

import javax.jws.WebMethod;
import javax.jws.WebService;
import javax.jws.soap.SOAPBinding;


import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.StringReader;
import java.util.UUID;
import java.util.regex.Pattern;

@WebService(serviceName = "HmmerObjects",
		portName = "HmmerObjectsPort",	
		targetNamespace = "http://hmmerobj.worker")
	

@SOAPBinding(style=SOAPBinding.Style.DOCUMENT,use=SOAPBinding.Use.LITERAL,parameterStyle=SOAPBinding.ParameterStyle.WRAPPED)
public class HmmerObjectsImpl {

	private final static String HEADER_END = "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -";
	private final static String NO_HITS    = "\t[no hits above thresholds]";
	private final static String SEQ_END    = "//";

	// Precompiled patterns and matcher strings
	private final static Pattern pDomain = Pattern.compile("\\d+/\\d+");
	private final static Pattern pNewHit = Pattern.compile(".+: domain .+");
	private final static String pFrom = ", from ";
	private final static String pOutput = "\t[output ";


	@WebMethod
	public String scoreRatingSameDB(String resultFile1, String resultFile2) throws Exception {
		
		System.out.println("Merging results of the same database:");
		int end_of_header=resultFile2.indexOf(HEADER_END);
		if (end_of_header>-1){
			resultFile1.concat(resultFile2.substring(end_of_header+HEADER_END.length()));
		}
		return resultFile1;
	}


	@WebMethod
        public String scoreRatingSameSeq(  String resultString1,
                                            String resultString2,
                                            int aLimit) throws Exception {
                StringBuilder result= new StringBuilder();

		// Write the header of the first file to the output file
		BufferedReader file1BR = new BufferedReader(new StringReader(resultString1));
		String line1;
		while (!(line1 = file1BR.readLine()).equals(HEADER_END)) result.append(line1+"\n");
		result.append(HEADER_END+"\n");
		
		// Skip the header of the second file
		BufferedReader file2BR = new BufferedReader(new StringReader(resultString2));
		String line2 = file2BR.readLine();
		while (!line2.equals(HEADER_END)) line2 = file2BR.readLine();
		
		/* Now iterate over the results for each sequence, merging them in the output file.
		 * Each iteration compares the hits for one sequence.
		 * Note that the same sequences are reported in both input files, even if they have
		 * no hits above thresholds.
		 */
		boolean merged = false;
		while (!merged) {
			/* We have to skip, for both input files:
			 * 
			 * Query sequence: UPI00000001F4
			 * Accession:      [none]
			 * Description:    [none]

			 * Scores for sequence family classification (score includes all domains):
			 * Model            Description                            Score    E-value  N
			 * --------         -----------                            -----    ------- ---
			 */
			while (!(line1 = file1BR.readLine()).startsWith("-")) result.append(line1+"\n");
			result.append(line1+"\n");
			while (!file2BR.readLine().startsWith("-")) {}
			
			/* Section 1.
			 * At each iteration, compare a hit from each result file, and write to the output
			 * file the hit with lowest e-value:
			 * SM00121           desc1                -52.2        2.7   1
			 * SM00478           desc2                -66.4        2.8   1
			 * ...
			 */
			line1 = file1BR.readLine();
			line2 = file2BR.readLine();
			boolean f1FirstSecEnd = (line1.charAt(0) == NO_HITS.charAt(0));
			boolean f2FirstSecEnd = (line2.charAt(0) == NO_HITS.charAt(0));
			boolean firstSecEnd = f1FirstSecEnd && f2FirstSecEnd;
			if (firstSecEnd) result.append(NO_HITS+"\n");
			while (!firstSecEnd) {
				if (f1FirstSecEnd) {
					result.append(line2+"\n");
					line2 = file2BR.readLine();
					firstSecEnd = line2.equals("");
				}
				else if (f2FirstSecEnd) {
					result.append(line1+"\n");
					line1 = file1BR.readLine();
					firstSecEnd = line1.equals("");
				}
				else { // we have two hits to compare
					double eValue1 = getEValue(line1),
						   eValue2 = getEValue(line2);
					if (eValue1 > eValue2) {
						result.append(line2+"\n");
						line2 = file2BR.readLine();
						f2FirstSecEnd = line2.equals("");
					}else {
						result.append(line1+"\n");
						line1 = file1BR.readLine();
						f1FirstSecEnd = line1.equals("");
						if (eValue1 == eValue2) {
							result.append(line2+"\n");
							line2 = file2BR.readLine();
							f2FirstSecEnd = line2.equals("");
							firstSecEnd = f1FirstSecEnd && f2FirstSecEnd;
						}
					}
				}
			}
			result.append("\n");
			
			// Skip blank if not already done
			if ((line1 = file1BR.readLine()).equals("")) line1 = file1BR.readLine();
			if (file2BR.readLine().equals("")) file2BR.readLine();
			result.append(line1+"\n");
			
			/* We have to skip, for both input files: 
			 * Parsed for domains:
			 * Model            Domain  seq-f seq-t    hmm-f hmm-t      score  E-value
			 * --------         ------- ----- -----    ----- -----      -----  -------
			 */
			while (!(line1 = file1BR.readLine()).startsWith("-")) result.append(line1+"\n");
			result.append(line1+"\n");
			while (!file2BR.readLine().startsWith("-")) {}
			
			/* Section 2.
			 * At each iteration, compare a hit from each result file, and write to the output
			 * file the hit with lowest seq-f (first position of the domain in the sequence):
			 * SM00235         1/1       1   109 [.     1   204 []   -68.9       14
			 * PF00041.11.ls   1/6     437   522 ..     1    84 []    49.3  1.5e-15
			 * PF00041.11.ls   2/6     825   914 ..     1    84 []    14.3  6.5e-07
			 * ...
			 */
			line1 = file1BR.readLine();
			line2 = file2BR.readLine();
			boolean f1SecondSecEnd = (line1.charAt(0) == NO_HITS.charAt(0));
			boolean f2SecondSecEnd = (line2.charAt(0) == NO_HITS.charAt(0));
			boolean secondSecEnd = f1SecondSecEnd && f2SecondSecEnd;
			if (f1SecondSecEnd)	file1BR.readLine(); // skip '//' or ' '
			if (f2SecondSecEnd)	file2BR.readLine(); // skip '//' or ' '
			String endSec = (aLimit == 0) ? "//" : "";
			if (secondSecEnd) result.append(NO_HITS+"\n");
			while (!secondSecEnd) {
				if (f1SecondSecEnd) {
					result.append(line2+"\n");
					line2 = file2BR.readLine();
					secondSecEnd = line2.equals(endSec);
				}else if (f2SecondSecEnd) {
					result.append(line1+"\n");
					line1 = file1BR.readLine();
					secondSecEnd = line1.equals(endSec);
				}else { // we have two hits to compare
					int seqF1 = getSeqF2nd(line1),
						seqF2 = getSeqF2nd(line2);
					if (seqF1 > seqF2) {
						result.append(line2+"\n");
						line2 = file2BR.readLine();
						f2SecondSecEnd = line2.equals(endSec);
					}else {
						result.append(line1+"\n");
						line1 = file1BR.readLine();
						f1SecondSecEnd = line1.equals(endSec);
						if (seqF1 == seqF2) {
							result.append(line2+"\n");
							line2 = file2BR.readLine();
							f2SecondSecEnd = line2.equals(endSec);
							secondSecEnd = f1SecondSecEnd && f2SecondSecEnd;
						}
					}
				}
			}
			result.append(endSec+"\n"); // new line or //
			
			/* Section 3.
			 * There are 3 cases here:
			 * -> A = 0: the 3rd section must not exist, we have already finished merging the results
			 * for the current sequence.
			 * -> A < 0: the 3rd section must exist, but no A was specified by the user. We must print
			 * all the results from both input files to the output file.
			 * -> A = N > 0: the 3rd section must exist, and an A value was specified by the user.
			 * We must print to the output the data corresponding to the top N results printed in section 2.
			 */
			if (aLimit != 0) {
				/* We have to skip for both input files, and also print to the output file:
				 * Alignments of top-scoring domains:
				 */
				if ((line1 = file1BR.readLine()).equals("")) line1 = file1BR.readLine();
				if (file2BR.readLine().equals("")) file2BR.readLine();
				result.append(line1+"\n");
				
				/* At each iteration, compare a hit from each result file, and write to the output
				 * file the hit with lowest seq-f (first position of the domain in the sequence):
				 * PF00041.11.ls: domain 1 of 6, from 437 to 522: score 49.3, E = 1.5e-15
                 * 					  *->P.saPtnltvtdvtstsltlsWspPt.gngpitgYevtyRqpkngge
                 *     				     P saP   + +++ ++ l ++W p +  ngpi+gY++++ +++ g+
   				 *		7LES_DROME   437    PiSAPVIEHLMGLDDSHLAVHWHPGRfTNGPIEGYRLRL-SSSEGNA 482
				 *
                 * 					  wkeltvpgtttsytltgLkPgteYtvrVqAvnggG.GpeS<-*
                 * 					  + e+ vp    sy+++ L++gt+Yt+ +  +n +G+Gp
  				 *		7LES_DROME   483 TSEQLVPAGRGSYIFSQLQAGTNYTLALSMINKQGeGPVA    522
				 *
				 * PF00041.11.ls: domain 2 of 6, from 825 to 914: score 14.3, E = 6.5e-07
                 * ...
				 */
				line1 = file1BR.readLine();
				line2 = file2BR.readLine();
				boolean f1ThirdSecEnd = (line1.charAt(0) == NO_HITS.charAt(0));
				boolean f2ThirdSecEnd = (line2.charAt(0) == NO_HITS.charAt(0));
				boolean thirdSecEnd = f1ThirdSecEnd && f2ThirdSecEnd;
				int numPrinted = 0;
				if (aLimit < 0) aLimit = Integer.MAX_VALUE;
				if (thirdSecEnd) result.append(NO_HITS+"\n");
				
				while (!thirdSecEnd && numPrinted < aLimit) {
					numPrinted++; // we will print at least one hit in this iteration
					if (f1ThirdSecEnd) {
						result.append(line2+"\n");
						thirdSecEnd = printAlignment(file2BR, result);
						if (!thirdSecEnd) line2 = file2BR.readLine();
					}
					else if (f2ThirdSecEnd) {
						result.append(line1+"\n");
						thirdSecEnd = printAlignment(file1BR, result);
						if (!thirdSecEnd) line1 = file1BR.readLine();
					}
					else { // we have two hits to compare
						int seqF1 = getSeqF3rd(line1),
							seqF2 = getSeqF3rd(line2);
						if (seqF1 > seqF2) {
							result.append(line2+"\n");
							f2ThirdSecEnd = printAlignment(file2BR, result);
							if (!f2ThirdSecEnd) line2 = file2BR.readLine();
						}
						else {
							result.append(line1+"\n");
							f1ThirdSecEnd = printAlignment(file1BR, result);
							if (!f1ThirdSecEnd) line1 = file1BR.readLine();
							if (seqF1 == seqF2 && numPrinted < aLimit) {
								result.append(line2+"\n");
								f2ThirdSecEnd = printAlignment(file2BR, result);
								if (!f2ThirdSecEnd) line2 = file2BR.readLine();
								thirdSecEnd = f1ThirdSecEnd && f2ThirdSecEnd;
								numPrinted++;
							}
						}
					}
				}
				if (numPrinted == aLimit)
					result.append("\t[output cut off at A = " + aLimit + " top alignments]"+"\n");
				
				// Ensure that both file pointers are situated at the beginning of the next sequence
				while (!file1BR.readLine().equals(SEQ_END)) { }
				while (!file2BR.readLine().equals(SEQ_END)) { }
				
				result.append(SEQ_END+"\n"); // end of current sequence
			}
			
			file1BR.mark(2);
			if (file1BR.read() >= 0)
				file1BR.reset();
			else
				merged = true;
		}
		
		file1BR.close();
		file2BR.close();

                return result.toString();
	}
	
	

	private static double getEValue(String hit) {
		int iniPos, fiPos;
		int i = hit.length() - 1;
		
		// Skip N
		while (hit.charAt(i) != ' ') i--;
		// Skip blank spaces
		while (hit.charAt(i) == ' ') i--;
		
		fiPos = i + 1;
		while (hit.charAt(i) != ' ') i--;
		iniPos = i + 1;
		
		return Double.parseDouble(hit.substring(iniPos, fiPos));
	}
	
	

	private static int getSeqF2nd(String hit) {
		int iniPos, fiPos;
		int i = 0;
		boolean atSeqF = false;
		while (!atSeqF) {
                    	StringBuilder sb = new StringBuilder();

			char c = hit.charAt(i);
                        while (c != ' ') {
                            	sb.append(c);
				c = hit.charAt(++i);
                        }
			if (pDomain.matcher(sb.toString()).matches())
				atSeqF = true;
                            while (hit.charAt(i) == ' ') i++;
		}
		iniPos = i;
                while (hit.charAt(i) != ' ') i++;
		fiPos = i;
                return Integer.parseInt(hit.substring(iniPos, fiPos));
                
	}
	
	
	private static int getSeqF3rd(String hit) {
		int iniPos, fiPos, i;
		iniPos = hit.indexOf(pFrom) + pFrom.length();
		i = iniPos;
		while (hit.charAt(i) != ' ') i++;
		fiPos = i;
		return Integer.parseInt(hit.substring(iniPos, fiPos));
	}
	
	
	private static boolean printAlignment(BufferedReader br, StringBuilder sb) throws Exception {
		br.mark(200);
		String line = br.readLine();
		boolean isNewHit = pNewHit.matcher(line).matches();
		boolean isOutputCut = line.startsWith(pOutput);
		boolean isSeqEnd = line.equals(SEQ_END);
		
		while (!(isNewHit || isOutputCut || isSeqEnd)) {
			sb.append(line+"\n");
			br.mark(200);
			line = br.readLine();
			isNewHit = pNewHit.matcher(line).matches();
			isOutputCut = line.startsWith(pOutput);
			isSeqEnd = line.equals(SEQ_END);
		}
		
		if (isNewHit) {
			br.reset();
			return false;
		}else if (isOutputCut) {
			return true;
		}else { 
			br.reset();
			return true;
		}
	}

}
