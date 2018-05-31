#include <stdio.h>
#include <stdlib.h>
#include "Fragment.h"
#include "kmeans_io.h"

    Fragment::Fragment(int nObjs, int nCoords){
	init(nObjs, nCoords);
    }
    void Fragment::init(int nObjs, int nCoords){
        int i;
	numObjs = nObjs;
        numCoords = nCoords;
        objects    = (float**) malloc(numObjs * sizeof(float*));
        assert(objects != NULL);
        objects[0] = (float*)  malloc(numObjs * numCoords * sizeof(float));
        assert(objects[0] != NULL);
        for (i=1; i<numObjs; i++)
                objects[i] = objects[i-1] + numCoords;
    }
    
    void Fragment::init_file(char* filename, int isBinaryFile){
       
	 objects = file_read(isBinaryFile, filename, &numObjs, &numCoords);
    }


