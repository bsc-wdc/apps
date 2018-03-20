#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */


void fillMatrix(const char* fileName, int matrixSize, double* mat) {
    int i, j;
    
    FILE *file;
    file = fopen(fileName, "r");

    // printf("  - Open file %s with size %d.\n", fileName, matrixSize);
    for(i = 0; i < matrixSize; i++) {
        for(j = 0; j < matrixSize; j++) {
            if (!fscanf(file, "%lf", &mat[i*matrixSize + j])) {
                break;
            }
            // printf("%lf\n", mat[i*matrixSize + j]);
        }
    }
    fclose(file);    
}

void storeMatrix(const char* fileName, int matrixSize, const double* mat) {
    int i, j;

    FILE *file;
    file = fopen(fileName, "w");

    // printf("  - Store matrix in file %s with size %d.\n", fileName, matrixSize);
    for(i = 0; i < matrixSize; i++) {
        for(j = 0; j < matrixSize; j++) {
            fprintf(file, "%lf ", mat[i*matrixSize + j]);
            // printf("%lf\n", mat[i*matrixSize + j]);
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

void printMatrix(int matrixSize, const double* mat) {
    int i, j;
    
    printf("******************************************************\n");
    printf("Result Matrix:\n");
    for (i = 0; i < matrixSize; i++) {
        printf("\n"); 
        for (j = 0; j < matrixSize; j++) {
            printf("%6.2f   ", mat[i*matrixSize + j]);
        }
    }
    printf("\n******************************************************\n");
}

int main (int argc, char *argv[]) {    
    /**************************** Initialize ************************************/
    if (argc != 5) {
        // printf("Incorrect usage: matmul <matrixSize> <Ain> <Bin> <Cout>. Quitting...\n");
        exit(1);
    }
    int matrixSize = atoi(argv[1]);     // Number of rows/columns in matrix A B and C
    char* ain = argv[2];                // FileName of Ain
    char* bin = argv[3];                // FileName of Bin
    char* cout = argv[4];               // FileName of Cout
    
    double a[matrixSize*matrixSize];   // Matrix A to be multiplied
    double b[matrixSize*matrixSize];   // Matrix B to be multiplied
    double c[matrixSize*matrixSize];   // Result matrix C    
    
    // Initialize MPI env
    int mpiProcs;                       // Number of MPI Nodes
    int taskid;                         // Task identifier
    MPI_Status status;                  // Status variable for MPI communications
    MPI_Request send_request;           // For async calls

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiProcs);
    
    // Misc variables
    int i, j, k, dest, mtype, averow, offset;
    
    /**************************** initialize task ************************************/
    if (taskid == MASTER) {
        // printf("Matmul with %d MPI nodes.\n", mpiProcs);
    }
        
    // Initialize arrays
    // printf("Initialize matrixes.\n");
    fillMatrix(ain, matrixSize, a);
    fillMatrix(bin, matrixSize, b);
    fillMatrix(cout, matrixSize, c);
    averow = matrixSize/mpiProcs;
    offset = taskid*averow;
    
    /**************************** compute task ************************************/
    // Perform multiply accumulative
    // printf("Perform multiply accumulative on process %d with offset %d and averow %d\n", taskid, offset, averow);
    for (i = offset; i < offset + averow; i++) {
        for (k = 0; k < matrixSize; k++) {
            for (j = 0; j < matrixSize; j++) {
                c[i*matrixSize + j] = c[i*matrixSize + j] + a[i*matrixSize + k]*b[k*matrixSize + j];
            }
        }
    }
    
    // Send back result to master
    // printf("Send result back to master on process %d.\n", taskid);
    mtype = FROM_WORKER;
    MPI_Isend(&c[offset*matrixSize], averow*matrixSize, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &send_request);
    
    /**************************** master task ************************************/
    if (taskid == MASTER) {
        // Receive results from worker tasks
        // printf("Receive results.\n");
        mtype = FROM_WORKER;
        for (i = 0; i < mpiProcs; i++) {
            offset = i*averow;
            MPI_Recv(&c[offset*matrixSize], averow*matrixSize, MPI_DOUBLE, i, mtype, MPI_COMM_WORLD, &status);
            // printf("  - Received results from task %d\n", i);
        }
        
        // Print result
        // printMatrix(matrixSize, c);
        
        // Store result to file
        // printf("Store result matrix.\n");
        storeMatrix(cout, matrixSize, c);
        
        printf ("Done.\n");
    }
    
    
    /**************************** Finalize ************************************/
    MPI_Finalize();
}
