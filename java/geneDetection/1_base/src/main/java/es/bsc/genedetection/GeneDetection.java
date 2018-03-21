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
package es.bsc.genedetection;

import es.bsc.compss.types.annotations.Constraints;
import es.bsc.compss.types.annotations.Orchestration;

import java.util.HashMap;

import java.util.List;

import javax.jws.WebMethod;
import javax.jws.WebParam;
import javax.jws.WebService;

import auxiliar.genedetect.Formatting;

import core.genedetect.BL2GAnnotation;
import core.genedetect.BLASTText;
import core.genedetect.Bl2GAnnotations;
import core.genedetect.BlastIDs;
import core.genedetect.CommentedNASequence;
import core.genedetect.FASTA;
import core.genedetect.GeneDetectMethods;
import core.genedetect.GenewiseReport;
import core.genedetect.NemusObject;
import dummy.genedetect.core.BioTools.BioToolsPort.BioTools;
import es.bsc.genedetection.coreelements.GeneWise;

import java.io.File;

import java.lang.String;

@WebService
public class GeneDetection {
	private HashMap<String, String[]> genomes = new HashMap<String, String[]>();
	private HashMap<String, CommentedNASequence> genomesCNAProperties = new HashMap<String, CommentedNASequence>();
	private HashMap<String, FASTA> sequences = new HashMap<String, FASTA>();
	private HashMap<String, String[]> runs = new HashMap<String, String[]>();
	private HashMap<String, GenewiseReport> reports = new HashMap<String, GenewiseReport>();
	
	static{
    	/*if (!Checkings.checkAmIMaster("GeneDetection")){
    		System.exit(-1);
    	}
    	Checkings.checkConnectivity();*/
    }
    
    @Constraints(computingUnits = "2", memorySize = "2.0")
	@WebMethod
    @Orchestration
    public String detectGenes(@WebParam(name="genomeName") String genomeName, @WebParam(name="sequenceName") String sequenceName, @WebParam(name="runID") String runID, int alignments, int scores, float threshold){    
        
    	/* #############
         * ##### A #####
         * #############*/

    	String genomeNCBI, genomeCNA;
    	CommentedNASequence cnaProperties;
    	boolean newGenoma = false;

    	if (!genomes.containsKey(genomeName)){
    		// A.1 - Translate Genome DB to NCBI BLAST format
    		newGenoma = true;
    		genomeNCBI = genomeName + "_NCBI.zip";
    		GeneDetectMethods.runNCBIFormatdb(genomeName, genomeNCBI);

    		// A.2 - Translate Genome DB to Commented NA format
    		genomeCNA = genomeName + "_CNA";
    		cnaProperties = GeneDetectMethods.fromFastaToCommentedNASequence(genomeName, genomeCNA);
    	}else{
    		String[] str = genomes.get(genomeName);
    		if (str.length !=2){
    			System.err.println("Error genome files not found");
    			return null;
    		}
    		genomeNCBI = str[0];
    		genomeCNA = str[1];
    		cnaProperties = genomesCNAProperties.get(genomeName);
    	}
    	/* #############
         * ##### B ##### Get a list of similar sequences and format them #
         * #############
         */
        
        FASTA fastaSeq;
        if (!sequences.containsKey(sequenceName)){ 	
        	System.out.println("loading sequence");
        	fastaSeq = Formatting.loadSequenceFromFile(sequenceName, "");
        }else{
        	System.out.println("Getting");
        	fastaSeq = sequences.get(sequenceName);
        }
        if (fastaSeq ==null){
        	System.err.println("Error getting FASTA sequence");
        	return null;
        }
        BLASTText blastReport = GeneDetectMethods.runNCBIBlastp(fastaSeq, Formatting.generateNCBIBlastpParameters(alignments, scores, threshold));
        //System.out.println("Blast report" + blastReport.getContent().getValue());
        // Parse sequence ids from the hits of the BLAST report
        BlastIDs bIds = BioTools.Static.parseBlastIDs(blastReport);

        // For each sequence id, get the corresponding protein (aminoacid sequence) and convert it to FASTA format
        List<NemusObject> seqIds = bIds.getIds(); // ####### OE Synchronization
        int numSeqs = seqIds.size(); 
        System.out.println("Num of seqs to evaluate: " +numSeqs);
        
        FASTA[] fastaSeqs = new FASTA[numSeqs];
        int i= 0;
        //Transforms Generic Sequences to FASTA format
        for (NemusObject seqId : seqIds) {
            // Make next two calls stateful WS core
        	System.out.println("Call seqId " +seqId);
        	BioTools btService = new BioTools();
            btService.loadAminoAcidSequence(seqId);
            fastaSeqs[i++] = btService.fromGenericSequenceToFasta(seqId);
        }
        
        /* #############
         * ##### C ##### Get a list of the most relevant regions in the Genome DB, according to the protein sequences #
         * #############
         */
        // For each protein, run a BLAST against the Genome DB, searching for the best scoring genes of the Genome
        BLASTText[] blastResults = new BLASTText[numSeqs]; // From B
        for (i = 0; i < numSeqs; i++) {
           
        	blastResults[i] = GeneDetectMethods.runNCBIBlastAgainstDBFromFASTA(genomeNCBI, // From A.1
                    fastaSeqs[i], // From B
                    Formatting.generateNCBIBlastParameters(threshold));
        }

        // Merge the BLAST results
        for (int next = 1; next < numSeqs; next *= 2) {
            for (int result = 0; result < numSeqs; result += 2 * next) {
                if (result + next < numSeqs) {
                    blastResults[result] = GeneDetectMethods.mergeBlastResults(blastResults[result], blastResults[result + next]);
                }
            }
        }

        // Pick the most relevant genes (regions) revealed by BLAST

        Bl2GAnnotations bl2gAnnots = BioTools.Static.runBlast2GeneWUniref90(blastResults[0], Formatting.generateBlast2GeneParameters());
        //Bl2GAnnotations bl2gAnnots = BioTools.Static.runBlast2Gene(blastResults[0], Formatting.generateBlast2GeneParameters(), Database.UNIREF_90);

        // Remove overlapped Genoma regions
        Bl2GAnnotations overlapAnnots = BioTools.Static.overlappingFromBL2G(bl2gAnnots);
        
        /* #############
         * ##### D ##### Run Genewise for each relevant region of the Genome DB and merge the results in a final report #
         * #############
         */
        // For each relevant region, run Genewise
        
        List<BL2GAnnotation> notOverlappedRegions = overlapAnnots.getAnnots(); // ####### OE Synchronization
        int numRegions = notOverlappedRegions.size();
        GenewiseReport[] gwResults = new GenewiseReport[numRegions];
        i = 0;

        String seqFile = "seqFile"+runID+".out";
        String genomeFile = "genomeFile"+runID+".out";

        for (BL2GAnnotation region : notOverlappedRegions) {
        	FASTA seq = getSequence(region.getProtID().getValue(), fastaSeqs);
            if (seq!=null){
            	String args = GeneDetectMethods.prepareGenewiseFiles(genomeCNA, cnaProperties, region, seq, seqFile, genomeFile);
            	//INSERT runGeneWise call
            	String rawReport = GeneWise.runGeneWise(seqFile, genomeFile, args);
            	gwResults[i++] = GeneDetectMethods.postProcessGenewise(rawReport, region);
            }
        }
        if (i<numRegions){
        	numRegions = i;
        }
        // Now merge the Genewise results
        for (int next = 1; next < numRegions; next *= 2) {
            for (int result = 0; result < numRegions; result += 2 * next) {
                if (result + next < numRegions) {
                    gwResults[result].mergeGenewiseResults(gwResults[result + next]);
                }
            }
        }
        GeneDetectMethods.bindResultsToSecureData(gwResults[0], runID);
        // Print the final report
        System.out.println("REPORT FROM GENEWISE:");
        System.out.println(gwResults[0].getGff().getContent().getValue());
        if (newGenoma)
        	runs.put(runID, new String[]{"/storage/"+genomeName+".nin","/storage/"+genomeName+".nsq","/storage/"+genomeName+".nhr", genomeNCBI, genomeCNA, seqFile, genomeFile});
        else
        	runs.put(runID, new String[]{seqFile, genomeFile});
        reports.put(runID, gwResults[0]);
        return gwResults[0].getGff().getContent().getValue();
    }
    
    @WebMethod
    @Orchestration
    public void removeRun(@WebParam(name="runID") String runID){    
		String[] str = runs.get(runID);
		if (str!=null){
			for (String s:str){
				 System.out.println("Remove file "+ s);
				File f = new File(s);
				f.delete();
			}
		}
		GeneDetectMethods.removeResultFromSecureData(runID);
		runs.remove(runID);
		reports.remove(runID);
    }
    
    @WebMethod
    public String getRunResults(@WebParam(name="runID") String runID){    
    	GenewiseReport gwResult = reports.get(runID);
        return gwResult.getGff().getContent().getValue();
    }
    
    @WebMethod
    public String[] getPreviousRuns(){
    	return runs.keySet().toArray(new String[runs.size()]);
    }
    
	@WebMethod
    @Orchestration
    public String loadGenoma(@WebParam(name="genomeName") String genomeName, @WebParam(name="location") String location){    
 
        // A.1 - Translate Genome DB to NCBI BLAST format
        String genomeNCBI = genomeName + "_NCBI.zip";
        GeneDetectMethods.runNCBIFormatdb(genomeName, genomeNCBI);

        // A.2 - Translate Genome DB to Commented NA format
        String genomeCNA = genomeName + "_CNA";
        CommentedNASequence cnaProperties = GeneDetectMethods.fromFastaToCommentedNASequence(genomeName, genomeCNA);
        
        genomes.put(genomeName, new String[]{genomeNCBI, genomeCNA});
        genomesCNAProperties.put(genomeName, cnaProperties);
        return genomeName;
    }
	
	@WebMethod
    @Orchestration
    public void removeGenoma(@WebParam(name="genomeName") String genomeName){    
        
		String[] str = genomes.get(genomeName);
		if (str != null){
			for (String s:str){
				File f = new File(s);
				f.delete();
			}
		}	
        genomes.remove(genomeName);
        genomesCNAProperties.remove(genomeName);
    }
	
	@WebMethod
    @Orchestration
    public String loadSequence(@WebParam(name="sequenceName") String sequenceName, @WebParam(name="location") String location){    
		FASTA fastaSeq = Formatting.loadSequenceFromFile(sequenceName, location);
		GeneDetectMethods.bindSequenceToSecureData(fastaSeq);
		sequences.put(sequenceName, fastaSeq);
		return sequenceName;
    }
	
	@WebMethod
    @Orchestration
    public void removeSequence(@WebParam(name="sequenceName") String sequenceName){    
		GeneDetectMethods.removeSecureSequence(sequenceName);
		sequences.remove(sequenceName);
    }
    
    private static FASTA getSequence(String protId, FASTA[] fastaSeqs) {
        for (FASTA sequence : fastaSeqs) {
        	System.out.println("**** ProtID: " + protId + "NemusId:"+ sequence.getNemusId());
            if (sequence.getNemusId().contains(protId)) // ####### OE Synchronization
            {
                return sequence;
            }
        }
        System.out.println("Returning null");
        return null;
    }
    
    
}
