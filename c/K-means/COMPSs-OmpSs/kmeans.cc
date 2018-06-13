/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c  (sequential version)                        */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

// Copyright (c) 2005 Wei-keng Liao
// Copyright (c) 2011 Serban Giuroiu
// Copyright (c) 2017 Jorge Ejarque // COMPSs/OmpSs porting
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// -----------------------------------------------------------------------------

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

/*---< usage() >------------------------------------------------------------*/
static void usage(char *argv0, float threshold) {
    char *help =
        "Usage: %s [switches] -i filename \n"
        "       -i filename    : file containing data to be clustered\n"
        "       -b             : input file is in binary format (default no)\n"
        "       -n num_clusters: number of clusters (K must > 1) (default 2)\n"
	"	-f num_frags   : number of fragments (must > 1) (default 2)\n"
        "       -l iterations   : number of fragments (must > 1) (default 10)\n"
        "       -t threshold   : threshold value (default %.4f)\n"
        "       -o             : output timing results (default no)\n";
    fprintf(stderr, help, argv0, threshold);
    exit(-1);
}


/*----< seq_kmeans() >-------------------------------------------------------*/
void kmeans( int numClusters, int numFrags, int loop_iteration, char* filePath, int isBinaryFile, int is_output_timing)
{
    int     numCoords, i, j, index, loop=0;
    double  timing, io_timing, clustering_timing;

    if (is_output_timing) io_timing = wtime();

    numCoords = file_read_coords(isBinaryFile, filePath);
    if (numCoords<1){
        fprintf(stderr,"Error reading number of coordinates");
        exit(-1);
    }
    cout << "numcoords: " << numCoords << endl;

    Clusters *clusters = new Clusters(numClusters, numCoords);
    for (i=0; i<numClusters; i++){
        printf("Initial cluster %i is [",i);
        for (j=0; j<numCoords; j++){
            clusters->coords[i][j] = (float) ((rand()) / (float)((RAND_MAX/10))-5);
            printf(" %.2f", clusters->coords[i][j]);
        }
        printf("]\n");
    }

    /* need to initialize clusters newClusterSize and newClusters */
    ClustersSize newClusterSize[numFrags];
    Clusters newCluster[numFrags];
    Fragment fragments[numFrags];
    
    compss_on();
    //sleep(10);
    if (is_output_timing) io_timing = wtime(); 
    //Init fragments
    for (i=0; i<numFrags; i++){
	printf("init fragment %i\n", i);
        init_Fragment(&fragments[i], filePath, isBinaryFile);
//        compss_wait_on(fragments[i]);
    }
//    compss_barrier();
    
//    compss_wait_on(*fragments);
/*
    for (int i = 0; i < numFrags; i++){
        compss_wait_on(fragments[i]);
    }
*/
/*
    Clusters *clusters = new Clusters(numClusters, numCoords);
    for (i=0; i<numClusters; i++){
        printf("Initial cluster %i is [",i);
        for (j=0; j<numCoords; j++){
            clusters->coords[i][j] = fragments[0].objects[i][j]; 
            printf(" %.2f", clusters->coords[i][j]);
        }
        printf("]\n");
    }
*/
    if (is_output_timing) {
        timing            = wtime();
        io_timing         = timing - io_timing;
        clustering_timing = timing;
    }

    do {
        for (i=0; i<numFrags; i++){
            
	        printf("Computing partial new cluster for fragment %i\n", i);
	        compute_newCluster(&fragments[i], clusters, &newCluster[i], &newClusterSize[i]);
	        //clusters->print();
            if (i>0){
		        printf("merging fragments %i\n", i);
                merge_newCluster(&newCluster[0], &newCluster[i], &newClusterSize[0], &newClusterSize[i]);
	        }
        }

	    update_clusters(&newCluster[0], &newClusterSize[0], clusters);

	    compss_wait_on(*clusters);
	    //clusters->print();
    } while (loop++ < loop_iteration);

    compss_barrier();

    clusters->print();

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


/*---< main() >-------------------------------------------------------------*/
int main(int argc, char **argv) {
    extern char   *optarg;
    extern int     optind;
           int     opt, i, j, isBinaryFile, is_output_timing, numClusters, numFrags, numObjs, loop_iteration;
//           int    *membership;    /* [numObjs] */
           char   *filename;
           float **objects;       /* [numObjs][numCoords] data objects */
           float **clusters;      /* [numClusters][numCoords] cluster center */
           float   threshold;
           double  timing, io_timing, clustering_timing;

    /* some default values */
    threshold        = 0.001;
    numClusters      = 2;
    numFrags         = 2;
    loop_iteration   = 10;
    isBinaryFile     = 0;
    is_output_timing = 0;
    filename         = NULL;

    while ( (opt=getopt(argc,argv,"p:i:n:f:l:t:abo"))!= EOF) {
        switch (opt) {
            case 'i': filename=optarg;
                      break;
            case 'b': isBinaryFile = 1;
                      break;
            case 't': threshold=atof(optarg);
                      break;
            case 'n': numClusters = atoi(optarg);
                      break;
            case 'f': numFrags = atoi(optarg);
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
    if (filename == 0 || numClusters < 1 || numFrags < 1 || loop_iteration < 1) usage(argv[0], threshold);

    kmeans(numClusters, numFrags, loop_iteration, filename, isBinaryFile, is_output_timing);

    return(0);
}
 
