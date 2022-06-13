#include "mpi.h"

#include <stdlib.h>
#include <stdio.h>


int main(int argc, char **argv) {
    //-------------------------------------------
    // INIT
    int myid, numprocs;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    //-------------------------------------------
    // PROCESS 0 IS THE ONLY ONE WORKING IN THIS EXAMPLE
    if (myid == 0) {
        // Process parameters: Output file, list of input files
        char* output_file = argv[1];
        int num_input_files = argc - 2;

        // Process input files
        for (int i = 0; i < num_input_files; ++i) {
            // Get name
            char* input_file = argv[2 + i];
            printf("Processing file %s\n", input_file);

            // Process content
            FILE* read_ptr = fopen(input_file, "r");
            if (read_ptr == NULL) {
                printf("Cannot open input file %s\n", input_file);
                exit(1);
            }

            int data_sum = 0;
            char* line = NULL;
            size_t len = 0;
            while (getline(&line, &len, read_ptr) != -1) {
                int num = atoi(line);
                data_sum = data_sum + num;
            }

            fclose(read_ptr);

            // Write content
            FILE* write_ptr = fopen(output_file, "a");
            if (write_ptr == NULL) {
                printf("Cannot open output file %s\n", output_file);
                exit(1);
            }

            fprintf(write_ptr, "%d\n", data_sum);

            fclose(write_ptr);
        }
    }

    //-------------------------------------------
    // FINISH
    MPI_Finalize();
    return 0;
}
