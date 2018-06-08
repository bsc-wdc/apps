# HMMER
HMMER is used for searching sequence databases for sequence homologs. It implements methods using probabilistic models called profile hidden Markov models (profile HMMs).

## Workflow
The COMPSs application contains three main blocks:
  *  **Split**: The application divides both, the database and the sequence file, into several partitions.
  *  **Hmmpfam**: The application searches for proteins in each database fragment that match the sequences within each sequences file fragment.  
  *  **ScoreRatingSameDB**: The application gathers the results obtained when comparing one sequences file fragment against all DB fragments 
  *  **ScoreRatingSameSeq**: The application gathers the results obtained for all the sequences file fragments

## Versions
  1.  **obj**: the hmmpfam and scoreRating stages are performed as method task
  2.  **ws**: the hmmpfam is a method CE but the scoreRating CEs run on a webservice
