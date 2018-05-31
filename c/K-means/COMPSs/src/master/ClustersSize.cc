#include <stdio.h>
#include <stdlib.h>
#include "ClustersSize.h"
    ClustersSize::ClustersSize(int nClusters){
	init(nClusters);
    }
    void ClustersSize::init(int nClusters){
        numClusters = nClusters;
	size = new int[numClusters];
	for (int i=0; i<numClusters;i++){
		size[i]=0;
	}
    }
