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

import es.bsc.compss.types.annotations.task.Method;
import es.bsc.compss.types.annotations.Parameter;
import es.bsc.compss.types.annotations.parameter.Direction;
import es.bsc.compss.types.annotations.parameter.Type;
import es.bsc.compss.types.annotations.task.Service;
import core.genedetect.FASTA;
import core.genedetect.BLASTText;
import core.genedetect.CommentedNASequence;
import core.genedetect.RunNCBIBlastAgainstDBFromFASTASecondaryParameters;
import core.genedetect.RunBlast2GeneSecondaryParameters;
import core.genedetect.Bl2GAnnotations;
import core.genedetect.RunNCBIBlastpSecondaryParameters;
import core.genedetect.BlastIDs;
import core.genedetect.NemusObject;
import core.genedetect.BL2GAnnotation;
import core.genedetect.GenewiseReport;
import es.bsc.compss.types.annotations.Constraints;
import java.lang.String;

public interface GeneDetectionItf {

	@Constraints(memorySize = "1.0", computingUnits = "1")
	@Method(declaringClass = "core.genedetect.GeneDetectMethods")
	public void runNCBIFormatdb(
			@Parameter(direction = Direction.IN, type = Type.STRING) String genome,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String genomeFile
	);

	@Constraints(memorySize = "0.5", storageSize = "10.0")
	@Method(declaringClass = "core.genedetect.GeneDetectMethods")
	public CommentedNASequence fromFastaToCommentedNASequence(
			@Parameter(direction = Direction.IN, type = Type.STRING) String genome,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String genomeFile
	);

	@Constraints(memorySize = "1.0", computingUnits = "1")
	@Method(declaringClass = "core.genedetect.GeneDetectMethods")
	public BLASTText runNCBIBlastAgainstDBFromFASTA(
			@Parameter(type = Type.FILE, direction = Direction.IN) String blastDBFile,
			@Parameter(direction = Direction.IN) FASTA fasta,
			@Parameter(direction = Direction.IN) RunNCBIBlastAgainstDBFromFASTASecondaryParameters params
	);

	@Constraints(memorySize = "0.5")
	@Method(declaringClass = "core.genedetect.GeneDetectMethods")
	public BLASTText mergeBlastResults(
			@Parameter(direction = Direction.IN) BLASTText res1,
			@Parameter(direction = Direction.IN) BLASTText res2
	);

	@Constraints(memorySize = "3.0", computingUnits = "1")
	@Method(declaringClass = "core.genedetect.GeneDetectMethods", isModifier = "true")
	public BLASTText runNCBIBlastp(
			@Parameter(direction = Direction.IN) FASTA fastaSeq,
			@Parameter(direction = Direction.IN) RunNCBIBlastpSecondaryParameters params
	);

	/*
	 * @Service(name = "BioTools", namespace = "http://genedetect.core", port =
	 * "BioToolsPort") public Bl2GAnnotations runBlast2Gene(@Parameter(direction
	 * = Direction.IN) BLASTText blastResult, @Parameter(direction =
	 * Direction.IN) RunBlast2GeneSecondaryParameters params,
	 * @Parameter(direction = Direction.IN) Database db);
	 */

	@Service(name = "BioTools", namespace = "http://genedetect.core", port = "BioToolsPort")
	public BlastIDs parseBlastIDs(
			@Parameter(direction = Direction.IN) BLASTText report
	);

	@Service(name = "BioTools", namespace = "http://genedetect.core", port = "BioToolsPort")
	public Bl2GAnnotations runBlast2GeneWUniref90(
			@Parameter(direction = Direction.IN) BLASTText blastResult,
			@Parameter(direction = Direction.IN) RunBlast2GeneSecondaryParameters params
	);

	@Service(name = "BioTools", namespace = "http://genedetect.core", port = "BioToolsPort")
	public Bl2GAnnotations overlappingFromBL2G(
			@Parameter(direction = Direction.IN) Bl2GAnnotations ovAnnots
	);

	@Service(name = "BioTools", namespace = "http://genedetect.core", port = "BioToolsPort")
	public void loadAminoAcidSequence(
			@Parameter(direction = Direction.IN) NemusObject seqId
	);

	@Service(name = "BioTools", namespace = "http://genedetect.core", port = "BioToolsPort")
	public FASTA fromGenericSequenceToFasta(
			@Parameter(direction = Direction.IN) NemusObject seqId
	);

	@Constraints(computingUnits = "1", memorySize = "0.5")
	@Method(declaringClass = "core.genedetect.GenewiseReport", isModifier = "true")
	public void mergeGenewiseResults(
			@Parameter(direction = Direction.IN) GenewiseReport report
	);

	@Method(declaringClass = "core.genedetect.GeneDetectMethods", isModifier = "true")
	public String prepareGenewiseFiles(
			@Parameter(type = Type.FILE, direction = Direction.IN) String genomeCNAFile,
			@Parameter(direction = Direction.IN) CommentedNASequence genomeDB,
			@Parameter(direction = Direction.IN) BL2GAnnotation region,
			@Parameter(direction = Direction.IN) FASTA sequence,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String seqFile,
			@Parameter(type = Type.FILE, direction = Direction.OUT) String genomeFile
	);

	@Method(declaringClass = "core.genedetect.GeneDetectMethods", isModifier = "true")
	public GenewiseReport postProcessGenewise(
			@Parameter(direction = Direction.IN) String rawReport,
			@Parameter(direction = Direction.IN) BL2GAnnotation region
	);

	@Constraints(computingUnits = "1", storageSize = "1.0", memorySize = "1.0")
	@Method(declaringClass = "es.bsc.genedetection.coreelements.GeneWise", isModifier = "true")
	public String runGeneWise(
			@Parameter(type = Type.FILE, direction = Direction.IN) String seqFile,
			@Parameter(type = Type.FILE, direction = Direction.IN) String genomeFile,
			@Parameter(direction = Direction.IN) String other_args
	);

	@Method(declaringClass = "core.genedetect.GeneDetectMethods", isModifier = "true")
	public void bindResultsToSecureData(
			@Parameter(direction = Direction.IN) GenewiseReport report,
			@Parameter(direction = Direction.IN, type = Type.STRING) String runID
	);

	@Method(declaringClass = "core.genedetect.GeneDetectMethods", isModifier = "true")
	public void bindSequenceToSecureData(
			@Parameter(direction = Direction.IN) FASTA fastaSeq
	);

	@Method(declaringClass = "core.genedetect.GeneDetectMethods", isModifier = "true")
	public void removeResultFromSecureData(
			@Parameter(direction = Direction.IN, type = Type.STRING) String runID
	);

	@Method(declaringClass = "core.genedetect.GeneDetectMethods", isModifier = "true")
	public void removeSecureSequence(
			@Parameter(direction = Direction.IN) String sequenceName
	);

}
