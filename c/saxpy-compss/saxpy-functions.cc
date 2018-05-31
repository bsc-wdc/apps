#include <stdio.h>
#include "saxpy.h"

void test(saxpy_obj *x){
    cout << "this is a test " << x->value[0] << endl;
}


void saxpy(int N, float a, saxpy_obj *y, saxpy_obj *x){

    for(int i = 0 ; i < N ; ++i){
        y->value[i] = a * x->value[i] + y->value[i];
    }
}
