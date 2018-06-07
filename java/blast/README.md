# BLAST
BLAST (Basic Local Alignment Search Tool) is an algorithm for comparing primary biological sequence information, such as the amino-acid sequences of different proteins or nucleotides of DNA sequences. BLAST enables a researcher to compare a query sequence with a library or database of sequences, and identify sequences that resemble the query sequence above a certain threshold.

## Workflow
The COMPSs application contains three main blocks:
  *  **Split**: the query sequences file is splitted in N fragments.
  *  **Alignment**: each sequence fragment is compared against the database by the blast binary.
  *  **Assembly**: assembly process combines all intermediate files into a single result file.

## Versions
  1.  **allone**: the assembly stage is performed sequentially on the master node
  2.  **tree**: the assembly stage is performed as a binary tree reduction with tasks 
  3.  **binary**: the application uses the @binary annotation to declare the CEs
