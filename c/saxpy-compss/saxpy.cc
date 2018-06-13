//#include <kernel.h>
#include <stdio.h>

#include "saxpy.h"

#define N 1024
#define REPEATS 1

int main(int argc, char* argv[]) {

    compss_on();

    float  a;
    saxpy_obj *x = new saxpy_obj();
    saxpy_obj *y = new saxpy_obj();
    float  la, lx[N], ly[N];

cout << "start init" << endl;
    x->init(N);
    y->init(N);   
cout << "end init" << endl; 


    a=5;
    int i;
    for (i=0; i<N; ++i){
        x->value[i]=i;
        y->value[i]=i+2;
        lx[i]=i;
        ly[i]=i+2;
    }

    for(int repeat = 0 ; repeat < REPEATS ; ++repeat){
		saxpy(N,a,y,x);
        const int n = N;
        for(i = 0 ; i < N ; ++i){
	    	ly[i] = a * lx[i] + ly[i];
        }
    }

    compss_wait_on(*y);

    compss_off();

    //Check results	
    for (i=0; i<N; ++i){
	if (y->value[i] != ly[i]){
	    printf("Error when checking results, in position %d\n",i);
	    printf("Expected %3e gets %3e\n",ly[i],y->value[i]);
	    return -1;
	}
    }
    printf("Results are correct\n");


    return 0;
}
