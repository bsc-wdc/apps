#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* getopt() */
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <c_compss.h>

#include <iostream>
#include <fstream>
using namespace std;

#include "cholesky.h"

void usage() {
    cout << "cholesky matrix_size block_size" << endl;
}

int main(int argc, char** argv) {

    if (argc != 3) {
        usage();
    }

    int MSIZE = atoi(argv[1]);
    int BSIZE = atoi(argv[2]);

    compss_on();
    
    struct timeval start_io, end_io, start_comp, end_comp;
    double io_timing, cholesky_timing;
    
    gettimeofday(&start_io, NULL);

    double** MATRIX = (double**)malloc(sizeof(double*)*MSIZE*MSIZE);
    
    for (int i = 0; i < MSIZE*MSIZE; ++i) {
      MATRIX[i] = (double*)malloc(sizeof(double)*BSIZE*BSIZE);
    }   
 
    //Hay MSIZE*MSIZE punteros a double. Es decir, MSIZE*MSIZE bloques.

   for (int i = 0; i < MSIZE; i++) {
      generate_block(MATRIX[i*MSIZE+i], BSIZE, 1);
      for (int j = i+1; j < MSIZE; j++) {
            //El bloque podria ser el mismo a asignar, pero siendo estrictos en el sentido
            //de no reservar mas memoria de la necesaria generamos el doble de tareas.
            generate_block(MATRIX[i*MSIZE+j], BSIZE, 0);
            generate_block(MATRIX[j*MSIZE+i], BSIZE, 0);
        }
    } 

    compss_barrier();

    gettimeofday(&end_io, NULL);

    io_timing = (end_io.tv_sec - start_io.tv_sec) * 1e6;
    io_timing = (io_timing + (end_io.tv_usec - start_io.tv_usec)) * 1e-6; 

    gettimeofday(&start_comp, NULL);

    //Llegados a este punto MATRIX contiene los apuntadores clave para generar las dependencias. 
    //Cada elemento de MATRIX es un puntero a un bloque, el bloque no esta en el nodo master, 
    //pero la direccion es lo unico necesario para ser capaces de crear dependencias entre tareas.
    
    for (int k = 0; k < MSIZE; ++k) {
        ddss_dpotrf(Lower, BSIZE, MATRIX[k*MSIZE+k], BSIZE); //La matriz es cuadrada

        for (int i = k+1; i < MSIZE; ++i) {
            ddss_dtrsm(Right, Lower, 
		               Trans, NonUnit, 
                	   BSIZE, BSIZE,
                       BSIZE, MATRIX[k*MSIZE+k], BSIZE,
                       MATRIX[i*MSIZE+k], BSIZE);
        }        

        for (int i = k+1; i < MSIZE; ++i) {

            ddss_dsyrk(Lower, NoTrans,
                       BSIZE, BSIZE,
                       BSIZE, MATRIX[i*MSIZE+k], BSIZE,
                      -BSIZE, MATRIX[i*MSIZE+i], BSIZE);
        
            for (int j = i; j < MSIZE; ++j) {

                ddss_dgemm(NoTrans, Trans,
                    	   BSIZE, BSIZE, BSIZE,
                           BSIZE, MATRIX[i*MSIZE+k], BSIZE,
                           MATRIX[j*MSIZE+k], BSIZE,
                           -BSIZE, MATRIX[i*MSIZE+j], BSIZE);
            }
        }
    }

    compss_barrier();

    gettimeofday(&end_comp, NULL);

    cholesky_timing = (end_comp.tv_sec - start_comp.tv_sec) * 1e6;
    cholesky_timing = (cholesky_timing + (end_comp.tv_usec - start_comp.tv_usec)) * 1e-6; 

    printf("\nPerforming **** Cholesky ****\n");
    printf("I/O time           = %10.4f sec\n", io_timing);
    printf("Computation timing = %10.4f sec\n", cholesky_timing);

    compss_off(); 
}
