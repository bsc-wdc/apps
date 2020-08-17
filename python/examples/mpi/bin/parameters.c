#include "mpi.h"

#include <stdio.h>
#include <stdlib.h>


int main(int argc, char** argv) {
    //-------------------------------------------
    // INIT
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    //-------------------------------------------
    // Process
    if (myid == 0) {
        int num_param = atoi(argv[1]);
        char* str_param = argv[2];
        char* file_param = argv[3];

        printf("Received parameters:\n");
        printf(" - Parameter 1: %d\n", num_param);
        printf(" - Parameter 2: %s\n", str_param);
        printf(" - Parameter 3: %s\n", file_param);

        printf(" - File content: \n");
        FILE* fptr = fopen(file_param, "r");
        if (fptr == NULL) {
            printf("Cannot open file \n");
            exit(1);
        }

        char c = fgetc(fptr);
        while (c != EOF) {
            printf("%c", c);
            c = fgetc(fptr);
        }

        fclose(fptr);
    }

    //-------------------------------------------
    // FINISH
    MPI_Finalize();
    return 0;
}
