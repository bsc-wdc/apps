# HMMER
HMMER is used for searching sequence databases for sequence homologs. It implements methods using probabilistic models called profile hidden Markov models (profile HMMs).

## Workflow
The COMPSs application contains three main blocks:
  *  **Split**: The application divides both, the database and the sequence file, into several partitions.
  *  **Hmmpfam**: The application searches for proteins in each database fragment that match the sequences within each sequences file fragment.  
  *  **ScoreRatingSameDB**: The application gathers the results obtained when comparing one sequences file fragment against all DB fragments 
  *  **ScoreRatingSameSeq**: The application gathers the results obtained for all the sequences file fragments

## OBJ Version
This version performs the hmmpfam as a method CE and scoreRating stages are performed as service tasks.

### Build Instructions ###
```
$ cd HMMER_WS_DIR/application
$ mvn clean package
```

### Execution Instructions ###
```
$ runcompss hmmerobj.HMMPfam <database> <sequences> <output> <numDBFrags> <numSeqFrags> <cmd_args>
  where:
    - database: Database Name
    - sequences: Input sequences path
    - output: Output file
    - numDBFrags: number of DB fragments
    - numSeqFrags: number of sequences fragments
    - cmd_args: Command line Arguments of the HMMER binary
``` 
### Execution Command Example ###
####TERMINAL 1: START SERVICE####
```
$ APPS_BASE/java/hmmer/2_ws/service/init.sh
```
####TERMINAL 2: RUN APPLICATION####
```
$ export HMMER_BINARY=APPS_BASE/java/hmmer/deps/binaries/hmmpfam

$ runcompss --classpath=APPS_BASE/java/hmmer/2_ws/application/target/hmmerws.jar --project=APPS_BASE/java/hmmer/2_ws/conf/project.xml --resources=APPS_BASE/java/hmmer/2_ws/conf/resources.xml hmmerws.HMMPfam  ${APPS_DIR}/datasets/Hmmer/smart.HMMs.bin ${APPS_DIR}/datasets/Hmmer/256seq /tmp/hmmer.result 4 4
```






