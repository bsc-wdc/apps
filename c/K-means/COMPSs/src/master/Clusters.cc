#include <stdio.h>
#include <stdlib.h>
#include <boost/serialization/serialization.hpp>    
#include "kmeans_io.h"
#include "Clusters.h"

    Clusters::Clusters(int nClusters, int nCoords){
	init(nClusters, nCoords);
    }

    void Clusters::init(int nClusters, int nCoords){
        int i; 
	numClusters = nClusters;
        numCoords = nCoords;
        coords    = (float**) malloc(numClusters * sizeof(float*));
        assert(coords != NULL);
        coords[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
        assert(coords[0] != NULL);
        for (i=1; i<numClusters; i++)
                coords[i] = coords[i-1] + numCoords;
    }
    
    Clusters::~Clusters(){
	if (coords[0] != NULL)
		free(coords[0]);
	if (coords != NULL)
		free(coords);
    }

