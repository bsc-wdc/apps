#ifndef CLUSTERSSIZE_H
#define CLUSTERSSIZE_H
#include <stdio.h>
#include <stdlib.h>
#include <boost/serialization/serialization.hpp>    
#include <boost/serialization/array.hpp>

using namespace boost;
using namespace serialization;

class ClustersSize 
{
public:
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar  & numClusters;
	if (Archive::is_loading::value)
        {
            if (size == NULL)
		size = new int[numClusters];
        }
	ar & make_array<int>(size, numClusters);
    }
    int numClusters;
    int* size;
    ClustersSize(){}
    ClustersSize(int nClusters);
    void init(int nClusters);
    ~ClustersSize(){
	if (size != NULL)
		free(size);
    }
};
#endif
