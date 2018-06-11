Usage:

    python generate_files NUM_POINTS NUM_CENTERS DIMENSIONS FRAGMENTS

    Generates N input files (where N=FRAGMENTS) for kmeans. NUM_POINTS points are randomly generated around NUM_CENTERS clusters, each of them having DIMENSIONS dimensions.

Example:
  
    python generate_files.py 2000 5 10 4

    This would generate the files N2000_K5_d10_0.txt, N2000_K5_d10_1.txt, N2000_K5_d10_2.txt and N2000_K5_d10_3.txt, each representing a different fragment.    
