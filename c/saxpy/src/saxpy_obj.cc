#include "saxpy_obj.h"


void saxpy_obj::init(int val){ 
    N = val;
    value = (float*) malloc(N*sizeof(float));
}

