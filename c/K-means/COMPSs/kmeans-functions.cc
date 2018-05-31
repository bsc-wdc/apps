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
#include "kmeans.h"


/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    float *coord1,   /* [numdims] */
                    float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;

    for (i=0; i<numdims; i++)
        ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
/*#pragma omp task*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   index, i;
    float dist, min_dist;

    /* find the cluster id that has min distance to object */
    index    = 0;
    min_dist = euclid_dist_2(numCoords, object, clusters[0]);

    for (i=1; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist) { /* find the min and its array index */
            min_dist = dist;
            index    = i;
        }
    }
    return(index);
}

void init_Fragment(Fragment *frag, char* filePath, int isBinaryFile){
        frag->init_file(filePath, isBinaryFile);
}


void update_clusters(Clusters *newClusters, ClustersSize *newClustersSize, Clusters *clusters){
	for (int i=0; i<(clusters->numClusters); i++){
            for (int j=0; j<(clusters->numCoords); j++) {
                if (newClustersSize->size[i] > 0){
                    clusters->coords[i][j] = newClusters->coords[i][j] / newClustersSize->size[i];
                }
            }
        }
}

void compute_newCluster(Fragment *frag, Clusters *clusters, Clusters *newClusters, ClustersSize *newClustersSize){
	int *index = new int[frag->numObjs]; 
	int i,j;
	clusters->print();
	newClusters->init(clusters->numClusters, clusters->numCoords);
        newClustersSize->init(clusters->numClusters);
	//printf("Computing cluster with %i element\n", frag.numObjs);
        for (i=0; i<frag->numObjs; i++) {
            /* find the array index of nestest cluster center */
            index[i] = find_nearest_cluster(clusters->numClusters, frag->numCoords, frag->objects[i],
                                         clusters->coords);
	    //printf("[obj %i: %i]->",i, index);
            /* update new cluster centers : sum of objects located within */
	}
	for (i=0; i<frag->numObjs; i++) {
            newClustersSize->size[index[i]]++;
	    for (j=0; j<frag->numCoords; j++){
                if (newClustersSize->size[index[i]]==1){
			newClusters->coords[index[i]][j]= 0.0;
		}
		newClusters->coords[index[i]][j] += frag->objects[i][j];
	    }
        }

}


void merge_newCluster(Clusters *newClusters_1, Clusters *newClusters_2, ClustersSize *newClustersSize_1, ClustersSize *newClustersSize_2){
        int i, j;
        for (i=0; i<newClusters_1->numClusters; i++){
                newClustersSize_1->size[i] = newClustersSize_1->size[i] + newClustersSize_2->size[i];
            for (j=0; j<newClusters_1->numCoords; j++) {
                newClusters_1->coords[i][j] = newClusters_1->coords[i][j] + newClusters_2->coords[i][j];
            }
        }
}

