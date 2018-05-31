#ifndef CLUSTERS_H
#define CLUSTERS_H
#include <stdio.h>
#include <stdlib.h>
#include <boost/serialization/serialization.hpp>    
#include <boost/serialization/array.hpp>

using namespace boost;
using namespace serialization;

class Clusters 
{
public:
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar  & numClusters;
        ar  & numCoords;
	if (Archive::is_loading::value)
        {
		printf("Generating all mallocs for clusters numClusters:%d numCoords:%d\n", numClusters, numCoords);
		fflush(NULL);
		int i;
		coords    = (float**) malloc(numClusters * sizeof(float*));
        	coords[0] = (float*)  malloc(numClusters * numCoords * sizeof(float));
        	for (i=1; i<numClusters; i++)
                	coords[i] = coords[i-1] + numCoords;
			
        }
	printf("Managing float array\n");
	fflush(NULL);
	ar & make_array<float>(coords[0], numClusters*numCoords);
    }
    int numClusters;
    int numCoords;
    float** coords;
    Clusters(){};
    Clusters(int nClusters, int nCoords);
    void init(int nClusters, int nCoords);
    ~Clusters();
};
#endif
