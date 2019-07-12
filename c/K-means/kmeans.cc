#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include "kmeans.h"
#include "kmeans_io.h"
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename \n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -k num_clusters: number of clusters (K must > 1) (default 2)\n"
        "	-n num_objs    : number of objects per fragment \n"
	"	-f num_frags   : number of fragments (must > 1) (default 2)\n"
        "       -d num_coords  : number of coordinates (must > 1) (default 2)\n"
        "       -l iterations   : number of fragments (must > 1) (default 10)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -o             : output timing results (default no)\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}
void kmeans( int numClusters, int numFrags, int objsFrag, int numCoords, int loop_iteration, char* filePath, int isBinaryFile, int is_output_timing)
{
    printf(" Execution with K=%d, Frags=%d, NpF=%d, d=%d, max_iters=%d\n", numClusters, numFrags, objsFrag, numCoords, loop_iteration);
 
    int  i, j, index, loop=1;
    double  timing, io_timing, clustering_timing;

    if (numCoords<1){
        fprintf(stderr,"Error reading number of coordinates");
        exit(-1);
    }
    srand(1000);

    
    float *clusters = (float*) malloc(numClusters*numCoords*sizeof(float));
    for (i=0; i<numClusters; i++){
        for (j=0; j<numCoords; j++){
            clusters[i*numCoords+j] = (float) ((rand()) / (float)((RAND_MAX))*2 - 1);
        }
    }

    //print_clusters(clusters, numClusters, numCoords);

    int **newClusterSize = (int **) malloc(numFrags*sizeof(int*));
    float **newClusters = (float **) malloc(numFrags*sizeof(float*));
    int **frag_index = (int **)malloc(numFrags*sizeof(int*));
    float **fragments = (float **)malloc(numFrags*sizeof(float*));
    for (j=0; j<numFrags; j++){
        frag_index[j] = (int*) malloc(sizeof(int));
        newClusters[j] = (float*) malloc(numClusters*numCoords*sizeof(float));
    	newClusterSize[j]=(int*) malloc(numClusters* sizeof(int));
        for (i=0; i<numClusters; i++){
		 newClusterSize[j][i]=0;
        }
    }
    compss_on();
    if (is_output_timing) io_timing = wtime();

    for (i=0; i<numFrags; i++){
        fragments[i] = init_Fragment(objsFrag, numCoords, objsFrag*numCoords, filePath, frag_index[i]);

    }
    compss_barrier();
    if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }

    do {
        for (i=0; i<numFrags; i++){
	    compute_newCluster(objsFrag, numCoords, numClusters, objsFrag*numCoords, numClusters*numCoords, frag_index[i], fragments[i], clusters, newClusters[i], newClusterSize[i]);

        }
        int neighbor = 1;
        while (neighbor < numFrags) {
            for (int f = 0; f < numFrags; f += 2 * neighbor) {
                if (f + neighbor < numFrags) {
                        //printf("Merging fragment %d with %d \n", f, f+neighbor);
        			merge_newCluster(numCoords, numClusters, numClusters*numCoords, newClusters[f], newClusters[f+neighbor], newClusterSize[f], newClusterSize[f+neighbor]);
                }
            }
            neighbor *= 2;
        }
        update_Clusters(numCoords, numClusters, numClusters*numCoords, clusters, newClusters[0], newClusterSize[0]);
    } while (loop++ < loop_iteration);

    compss_barrier();

    if (is_output_timing) {
        timing            = wtime();
        clustering_timing = timing - clustering_timing;
        printf("\nPerforming **** Regular Kmeans (compss version) ****\n");
        printf("Input file:     %s\n", filePath);
        printf("numFrags     = %d\n", numFrags);
        printf("numClusters   = %d\n", numClusters);
        printf("Loop iterations    = %d\n", loop_iteration);
        printf("I/O time           = %10.4f sec\n", io_timing);
        printf("Computation timing = %10.4f sec\n", clustering_timing);
    }
    compss_off();
}

int main(int argc, char **argv) {
    extern char   *optarg;
    extern int     optind;
           int     opt, i, j, isBinaryFile, is_output_timing, numClusters, numFrags, numObjs, numCoords, loop_iteration;
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
    threshold        = 0.001;
    numClusters      = 2;
    numFrags         = 2;
    numObjs	     = 100000;
    numCoords        = 2;
    loop_iteration   = 10;
    isBinaryFile     = 0;
    is_output_timing = 0;
    filename         = NULL;
    while ( (opt=getopt(argc,argv,"p:i:n:k:f:l:t:d:abo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'k': numClusters = atoi(optarg);
                      break;
            case 'n': numObjs = atoi(optarg);
                      break;
            case 'f': numFrags = atoi(optarg);
                      break;
            case 'd': numCoords = atoi(optarg);
                      break;
            case 'l': loop_iteration = atoi(optarg);
                      break;
            case 'o': is_output_timing = 1;
                      break;
            case '?': usage(argv[0], threshold);
                      break;
            default: usage(argv[0], threshold);
                      break;
        }
    }
    if (filename == 0 || numClusters <= 1 || numFrags < 1 || numCoords <1 || loop_iteration < 1) usage(argv[0], threshold);
    kmeans(numClusters, numFrags, numObjs, numCoords, loop_iteration, filename, isBinaryFile, is_output_timing);
    return(0);
}
