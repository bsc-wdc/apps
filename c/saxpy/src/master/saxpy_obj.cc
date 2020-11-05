#include "saxpy_obj.h"


void Saxpyobj::init(int N){ 
    value = (float*) aligned_alloc(sizeof(float)*N,sizeof(float)*N);
}

