#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

using namespace std;

class saxpy_obj {

public:
    float *value;

    saxpy_obj(){};

    void init(int N);

};
