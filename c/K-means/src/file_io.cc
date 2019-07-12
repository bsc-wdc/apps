#include <stdio.h>
#include <stdlib.h>
#include <string.h>     /* strtok() */
#include <sys/types.h>  /* open() */
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>     /* read(), close() */
#include "kmeans_io.h"
#define MAX_CHAR_PER_POINT 128
float* file_read(int   isBinaryFile,  /* flag: 0 or 1 */
                  char *filename,
		  int  numObjs,       /* no. data objects (local) */
    		  int  numCoords)
{
    float *objects;
    int     i, j, len;
    ssize_t numBytesRead;
    if (isBinaryFile) {  /* input file is in raw binary format -------------*/
        int infile;
        if ((infile = open(filename, O_RDONLY, "0600")) == -1) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }
        printf("File %s numObjs   = %d\n",filename,numObjs);
        printf("File %s numCoords = %d\n",filename,numCoords);
        len = numObjs * numCoords;
        objects = (float*) malloc(len * sizeof(float));
        numBytesRead = read(infile, objects, len*sizeof(float));
        close(infile);
    }
    else {  /* input file is in ASCII format -------------------------------*/
        FILE *infile;
        char *line, *ret;
        int   lineLen;
        if ((infile = fopen(filename, "r")) == NULL) {
            fprintf(stderr, "Error: no such file (%s)\n", filename);
            return NULL;
        }
        lineLen = MAX_CHAR_PER_POINT*numCoords;
        line = (char*) malloc(lineLen);
        printf("File %s numObjs   = %d\n",filename,numObjs);
        printf("File %s numCoords = %d\n",filename,numCoords);
        len = numObjs * numCoords;
        objects = (float*) malloc(len * sizeof(float));
        i = 0;
        while (fgets(line, lineLen, infile) != NULL && i < numObjs) {
            char *saveptr;
	    if (strtok_r(line, " \t\n", &saveptr) == NULL) continue;
            for (j=0; j<numCoords; j++){
                char* tok = strtok_r(NULL, " ,\t\n", &saveptr);
		if (tok != NULL){
			objects[i*numCoords + j] = atof(tok);
		}else{
			objects[i*numCoords + j] = 0.0;
		}
	    }
            i++;
        }
        fclose(infile);
        free(line);
    }
    return objects;
}

int file_write(char      *filename,     /* input file name */
               int        numClusters,  /* no. clusters */
               int        numObjs,      /* no. data objects */
               int        numCoords,    /* no. coordinates (local) */
               float    **clusters,     /* [numClusters][numCoords] centers */
               int       *membership)   /* [numObjs] */
{
    FILE *fptr;
    int   i, j;
    char  outFileName[1024];
    sprintf(outFileName, "%s.cluster_centres", filename);
    printf("Writing coordinates of K=%d cluster centers to file \"%s\"\n",
           numClusters, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numClusters; i++) {
        fprintf(fptr, "%d ", i);
        for (j=0; j<numCoords; j++)
            fprintf(fptr, "%f ", clusters[i][j]);
        fprintf(fptr, "\n");
    }
    fclose(fptr);
    sprintf(outFileName, "%s.membership", filename);
    printf("Writing membership of N=%d data objects to file \"%s\"\n",
           numObjs, outFileName);
    fptr = fopen(outFileName, "w");
    for (i=0; i<numObjs; i++)
        fprintf(fptr, "%d %d\n", i, membership[i]);
    fclose(fptr);
    return 1;
}

void print_clusters(float *clusters, int numClusters, int numCoords) {
    int i, j;
            for (i=0; i < numClusters; ++i) {
            printf("Cluster %d [ ", i);
            for (j=0; j < numCoords; ++j) {
                printf("%f ", clusters[i*numCoords+j]);
            }
            printf("]\n");
        }
}
