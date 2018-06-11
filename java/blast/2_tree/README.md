# BLAST
BLAST (Basic Local Alignment Search Tool) is an algorithm for comparing primary biological sequence information, such as the amino-acid sequences of different proteins or nucleotides of DNA sequences. BLAST enables a researcher to compare a query sequence with a library or database of sequences, and identify sequences that resemble the query sequence above a certain threshold.

## Workflow
The COMPSs application contains three main blocks:
  *  **Split**: the query sequences file is splitted in N fragments.
  *  **Alignment**: each sequence fragment is compared against the database by the blast binary.
  *  **Assembly**: assembly process combines all intermediate files into a single result file.

## ALL ONE Version
This version performs the assembly stage of the workflow as a single task.

### Build Instructions ###
```
$ cd BLAST_TREE_DIR
$ mvn clean package
```

### Execution Instructions ###
```
$ runcompss blast.Blast <debug> <binary> <database> <sequences> <#fragments> <tmpdir> <output> <cmd_args>
  where:
    - debug: Set the debug mode on
    - binary: Blast binary location
    - database: Database Name
    - sequences: Input sequences path
    - #fragments: Fragments number
    - tmpdir: Temporary directory (must end with /)
    - output: Output file
    - cmd_args: Command line Arguments of the Blast binary
``` 
### Execution Command Example ###
```
$ runcompss --classpath=APPS_BASE/java/blast/2_tree/target/blasttree.jar blast.Blast true APPS_BASE/java/blast/deps/binaries/blastall APPS_BASE/datasets/Blast/databases/swissprot/swissprot APPS_BASE/datasets/Blast/sequences/sargasso_test.fasta 4 /tmp/ /tmp/result.txt
```






