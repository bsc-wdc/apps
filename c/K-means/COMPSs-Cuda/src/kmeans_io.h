#ifndef _H_KMEANS
#define _H_KMEANS

#include <assert.h>

#define msg(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); } while (0)
#define err(format, ...) do { fprintf(stderr, format, ##__VA_ARGS__); exit(1); } while (0)

double  wtime(void);
int file_read_coords(int, char*);
float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);

#endif
