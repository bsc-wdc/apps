#include <assert.h>

double  wtime(void);
int file_read_coords(int, char*);
float** file_read(int, char*, int*, int*);
int     file_write(char*, int, int, int, float**, int*);
