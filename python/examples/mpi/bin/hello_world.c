#include "mpi.h"

#include <stdio.h>


int main(int argc, char** argv) {
    //-------------------------------------------
    // INIT
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    //-------------------------------------------
    // Print
    printf("Hello world from processor %d out of %d processors\n", myid, numprocs);

    //-------------------------------------------
    // FINISH
    MPI_Finalize();
    return 0;
}
