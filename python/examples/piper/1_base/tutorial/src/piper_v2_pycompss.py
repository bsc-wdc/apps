#
#  Copyright 2002-2015 Barcelona Supercomputing Center (www.bsc.es)
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


# First translation of 'piper.nf', without error checking

import os
import shutil
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from pycompss.api.task import task
from pycompss.api.parameter import *

#@task (filename=FILE_OUT)
def createOutFile(filename):
  open(filename, "a+").close()

#@task (directory=FILE_OUT)
def createDirectory(directory):
  if not os.path.exists(directory): os.makedirs(directory)

#@task (current_file_name=FILE_OUT)
def writeSequence(record, current_file_name):
  current_file = open(current_file_name, "a+")
  SeqIO.write(record, current_file, "fasta")
  current_file.close()

#@task (directory=FILE_IN, current_file_name=FILE_OUT)
def writeSequenceInDirectory(record, directory, current_file_name):
  currentFile = os.path.join(directory, current_file_name)
  current_file = open(currentFile, "a+")
  SeqIO.write(record, currentFile, "fasta")
  current_file.close()

def traverseGenomesDirectory(genomes_folder):
  result = {}
  
  for subdir, dirs, files in os.walk(genomes_folder):
    # print "Current Subdir "+subdir

    for file in files:
      if file.endswith(".fa"):
        # The subdirectory name is the genome name
        result[os.path.split(subdir)[-1]] = os.path.join(subdir,file)
  
  return result


@task(genomeFile=FILE_IN, blastDBPackage = FILE_OUT)
def formatBlast(genomeFile, blastDB, blastDBPackage, blastStrategy):
  
  if (blastStrategy == "ncbi-blast"):
    command = "makeblastdb -dbtype nucl -in "+genomeFile+" -out "+blastDB
    print command
    os.system(command)
  elif (blastStrategy == "wu-blast"):
    command = "xdformat -n -o "+blastDB+" "+genomeFile
    print command
    os.system(command)

  command = "tar cfj "+blastDBPackage+" "+blastDB+"*"
  os.system(command)
  
 
@task(blastDBPackage=FILE_IN, querySequence=FILE_IN, blastHits=FILE_OUT)
def blast(blastDBName, blastDBPackage, querySequence, blastHits, blastStrategy, numCPUs):

  command = "tar xfj "+blastDBPackage
  os.system(command)

  if ( blastStrategy == "ncbi-blast" ):
  
    fmt = '6 qseqid sseqid evalue score qgi bitscore length nident positive mismatch pident ppos qacc gaps gaopen qaccver qlen qframe qstart qend sframe sstart send'
    command = "blastn -db "+blastDBName+" -query "+querySequence+" -outfmt \""+fmt+"\" -num_threads "+str(numCPUs)+" -out "+blastHits
    print command
    os.system(command)
    
  elif( params.blastStrategy == "wu-blast" ):
    command = "wu-blastn "+blastDBName+" "+querySequence+" -mformat=2 -e 0.00001 -cpus "+str(numCPUs)+" -filter=seg -lcfilter -errors -novalidctxok -nonnegok > "+blastHits
    os.system(command)

@task(exonerateQuery=FILE_IN, blastResult=FILE_IN, gtfOut=FILE_OUT, fastaOut=FILE_OUT)
def exonerate(specie, exonerateQuery, blastResult, chr_db, gtfOut, fastaOut, exonerateMode, repeatCov):
  createDirectory(specie)
  for record in chr_db:
    writeSequenceInDirectory(record, specie, record.id)
  
  command = ("echo $SHELL \n"
    "exonerateRemapping.pl "
    "-query "+exonerateQuery+" "
    "-mf2 "+blastResult+" "
    "-targetGenomeFolder "+specie+" "
    "-exonerate_lines_mode "+exonerateMode+" "
    "-exonerate_success_mode "+exonerateMode+" "
    "-ner no "
    "\n"
    "ls *.gtf *.fa \n"
    "\n"
    "if [ -s "+specie+".fa ]; then \n"
    "  repeat.pl "+specie+".fa "+specie+".ex.gtf "+str(repeatCov)+" \n"
    "  [[ ! -s rep"+str(repeatCov)+".fa ]] && exit 0 \n"
    "  mv "+specie+".fa chunk.seq \n"
    "  mv "+specie+".ex.gtf chunk.ex.annot \n"
    "  mv rep"+str(repeatCov)+".fa "+specie+".fa \n"
    "  mv rep"+str(repeatCov)+".ex.gtf "+specie+".ex.gtf \n"
    "fi \n"
    "\n"
    "pwd \n"
    "ls -l \n"
    "mv `basename "+blastResult+".fa` "+fastaOut+" \n"
    "mv `basename "+blastResult+".ex.gtf` "+gtfOut+" \n"
    "ls -l" )
    
  #print command
  os.system(command)

@task(returns=int, exonerateFasta=FILE_IN, normalizedFasta=FILE_INOUT)
def normExonerate(genome, sequenceName, exonerateFasta, normalizedFasta, previousMatch):
  exonerateSequences = open(exonerateFasta, "r")
  match = False
  
  for exonerateSequence in SeqIO.parse(exonerateSequences, "fasta"):
    if (exonerateSequence.id.find(sequenceName) != -1):
      writeSequence(SeqRecord (exonerateSequence.seq, id=exonerateSequence.id+"_"+genome, description=""), normalizedFasta)
      match = True
      
  exonerateSequences.close()
  return previousMatch or match

@task(fastaFile=FILE_IN, alignedFile=FILE_OUT)
def align(fastaFile, alignedFile, alignStrategy, numCPUs):
  command = "t_coffee -in "+fastaFile+" -method "+alignStrategy+" -n_core "+str(numCPUs)+" -run_name=temporal.aln"
  print command
  os.system(command)
  command = "mv temporal.aln "+alignedFile
  os.system(command)

@task(alignedFastaFile=FILE_IN, fastaBaseName=FILE_OUT)
def similarity(alignedFastaFile, fastaBaseName):
  command = "t_coffee -other_pg seq_reformat -in "+alignedFastaFile+" -output sim > "+fastaBaseName
  print command
  os.system(command)

  
#@task(dataDir=FILE_IN, queryFile=FILE_IN, matrixFile=FILE_OUT)
def matrix(dataDir, queryFile, allGenomes, matrixFile):
  tmpGenomesDir = "tmp_genomes"
  createDirectory(tmpGenomesDir)
  for genome in allGenomes:
    createDirectory(os.path.join(tmpGenomesDir, genome))
    
  command = ("echo \\n====== Pipe-R sim matrix =======\n"
             "sim2matrix.pl -query "+queryFile+" -data_dir "+dataDir+" -genomes_dir "+tmpGenomesDir+" | tee "+matrixFile+" \n"
             "echo \\n")
  print command
  os.system(command)

#@task(outputFile=FILE_IN)
def moveOutputs(outputFile, resultDir):
  shutil.move(outputFile, resultDir)

def flushFile(fileDescriptor, outputFileName):
  with open(outputFileName, "w") as fileOut:
    for line in fileDescriptor:
      fileOut.write(line)

if __name__ == "__main__" :

  from pycompss.api.api import compss_open, compss_wait_on

  genomes_folder = "/home/cdiaz/CRG/PIPER/tutorial/genomes/"
  queryFile      = "/home/cdiaz/CRG/PIPER/tutorial/5_RNA_queries.fa"
  dbPath         = "db"
  blastStrategy  = "ncbi-blast"
  numCPUs        = 1
  exonerateMode  = "exhaustive"
  alignStrategy  = "slow_pair"
  repeatCov      = 20
  resultDir      = "/home/cdiaz/CRG/result"

  
  allGenomes   = traverseGenomesDirectory(genomes_folder)
  
  allDatabases = {}
  allChr       = {} 
  
  # Create the BLAST / ISOLATED SEQUENCES databases for all genomes found
  for genome in allGenomes:
  
    blastDBPath = os.path.join(dbPath, genome, blastStrategy+"-db")
    chrPath     = os.path.join(dbPath, genome, "chr")
  
    
    # blastDB     = os.path.join(blastDBPath, "db")
    blastDB = genome+"_blast_db"
    blastDBPackage = blastDB+".tar.bz2"
    
    formatBlast(allGenomes[genome], blastDB, blastDBPackage, blastStrategy)
    allDatabases[genome, "package"] = blastDBPackage
    allDatabases[genome, "db"] = blastDB
    
    # createDirectory(chrPath)
    allChr[genome] = []
    currentGenomeFile = open(allGenomes[genome], "r")
    for currentSequence in SeqIO.parse(currentGenomeFile, "fasta"):
      allChr[genome].append(currentSequence)
      # writeSequenceInDirectory(currentSequence, chrPath, currentSequence.id)
      

    
  allQuerySequences = []
  blastHits = {}
  
  for genome in allGenomes:
    #blastNSQ = allDatabases[genome]
    #blastHits[genome] = os.path.join("tmp", genome+"_hits")
    blastHits[genome] = genome+"_hits"
    blast(allDatabases[genome, "db"], allDatabases[genome, "package"], queryFile, blastHits[genome], blastStrategy, numCPUs)
  
  exonerateResults = {}
  for genome in allGenomes:
    exonerateResults[genome, "fasta"] = genome+"_hits.fa"
    exonerateResults[genome, "gtf"] = genome+"_hits.ex.gtf"
    exonerate(genome, queryFile, blastHits[genome], allChr[genome], exonerateResults[genome, "gtf"], exonerateResults[genome, "fasta"], exonerateMode, repeatCov)
 
  normalizedFastaFiles = []
  thereismatch         = {}
  querySequencesFile = open(queryFile, "r")
  for querySequence in SeqIO.parse(querySequencesFile, "fasta"):
    normalizedFasta = querySequence.id+".mfa"
    
    if os.path.exists(normalizedFasta): os.unlink(normalizedFasta)
    
    writeSequence(querySequence, normalizedFasta)
    thereismatch[normalizedFasta] = False
    for genome in allGenomes:
      thereismatch[normalizedFasta] = normExonerate(genome, querySequence.id, exonerateResults[genome, "fasta"], normalizedFasta, thereismatch[normalizedFasta])
  
  # Synchronization to verify the available files that hit on exonerate
  for normalizedFasta in thereismatch.keys():
    match = compss_wait_on(thereismatch[normalizedFasta])
    
#    print "NORMALIZED FASTA "+normalizedFasta+" IS ", match," ******************"

    if not match:
      os.unlink(normalizedFasta)
    else:
      normalizedFastaFiles.append(normalizedFasta)

  
  alignedFiles = []
  for fastaFile in normalizedFastaFiles:
    currentAlignedFile = os.path.splitext(fastaFile)[0] + ".aln"
    align(fastaFile, currentAlignedFile, alignStrategy, numCPUs)
    alignedFiles.append(currentAlignedFile) 
  
  
  
  createOutFile("simMatrix.csv")
  if os.path.exists("data"):
    shutil.rmtree("data")
  
  os.makedirs("data")
    
  print "Aligned files size = ",len(alignedFiles)
  finalDataFiles = []
  for alignedFasta in alignedFiles:
    currentData = os.path.splitext(alignedFasta)[0]
    finalDataFiles.append(currentData)
    similarity(alignedFasta, os.path.splitext(alignedFasta)[0])
  
  for outFileName in finalDataFiles:
    outFD = compss_open(outFileName)
    flushFile(outFD, os.path.join("data", outFileName))
  #moveOutputs(os.path.splitext(alignedFasta)[0], "data")
  
  matrix("data", queryFile, allGenomes.keys(), "simMatrix.csv")
  
  if os.path.exists(resultDir):
    shutil.rmtree(resultDir)
  
  os.makedirs(resultDir)

  moveOutputs("simMatrix.csv", resultDir)
  for genome in allGenomes:
    outGTF = compss_open(exonerateResults[genome, "gtf"])
    flushFile(outGTF, os.path.join(resultDir, exonerateResults[genome, "gtf"]))

  
