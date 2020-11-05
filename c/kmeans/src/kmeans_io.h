#ifndef _H_KMEANS
#define _H_KMEANS
float* file_read(int, char*, int, int);
int     file_write(char*, int, int, int, float**, int*);
void print_clusters(float *clusters, int numClusters, int numCoords);
double wtime(void);
#endif
