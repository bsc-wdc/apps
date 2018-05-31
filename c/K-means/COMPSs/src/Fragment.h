#ifndef FRAGMENT_H
#define FRAGMENT_H

#include <stdio.h>
#include <stdlib.h>
#include <boost/serialization/serialization.hpp>    
#include <boost/serialization/array.hpp>

using namespace boost;
using namespace serialization;

class Fragment 
{
public:
    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar  & numObjs;
        ar  & numCoords;
	if (Archive::is_loading::value)
        {
            //if (objects == NULL){
		int i;
        	objects = (float**) malloc(numObjs * sizeof(float*));
        	objects[0] = (float*)  malloc(numObjs * numCoords * sizeof(float));
        	for (i=1; i<numObjs; i++)
                	objects[i] = objects[i-1] + numCoords;
	    //}
        }
	ar & make_array<float>(objects[0], numObjs*numCoords);
    }
    int numObjs;
    int numCoords;
    float** objects;
    Fragment(){}
    Fragment(int nObjs, int nCoords);
    void init(int nObjs, int nCoords);
    void init_file(char* filename, int isBinaryFile);
    ~Fragment(){
	/*if (objects[0] != NULL)
		free(objects[0]);
	if (objects != NULL)
		free(objects);*/
    }
};
#endif
