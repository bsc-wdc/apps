# HMMER
HMMER is used for searching sequence databases for sequence homologs. It implements methods using probabilistic models called profile hidden Markov models (profile HMMs).

## Workflow
The COMPSs application contains three main blocks:
  *  **Split**: The application divides both, the database and the sequence file, into several partitions.
  *  **Hmmpfam**: The application searches for proteins in each database fragment that match the sequences within each sequences file fragment.  
  *  **ScoreRatingSameDB**: The application gathers the results obtained when comparing one sequences file fragment against all DB fragments 
  *  **ScoreRatingSameSeq**: The application gathers the results obtained for all the sequences file fragments

## OBJ Version
This version performs the hmmpfam and scoreRating stages are performed as method task.

### Build Instructions ###
```
$ cd HMMER_OBJ_DIR
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
```
$ export HMMER_BINARY=APPS_BASE/java/hmmer/deps/binaries/hmmpfam

$ runcompss --classpath=APPS_BASE/java/hmmer/1_obj/target/hmmerobj.jar hmmerobj.HMMPfam APPS_BASE/datasets/Hmmer/smart.HMMs.bin APPS_BASE/datasets/Hmmer/256seq /tmp/hmmer.result 4 4
```






