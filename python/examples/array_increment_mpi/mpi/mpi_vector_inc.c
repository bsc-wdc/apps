#include <stdio.h>
#include <stdlib.h>
#include <math.h> 
#include <mpi.h>
int MAX_VECTOR_STR=10000;
int MAX_ARRAY=100;
int loadArray( char* fileName, int* array){
   //int MAX_VECTOR_STR=10000;
   char str1[MAX_VECTOR_STR];
   FILE * fp;
   int array_size=0;
   fp = fopen (fileName, "r");
   if (fp){
	char c;
	c = fgetc(fp);
	//printf("character: %c",c);
	if (c=='['){
		int index=0;
		c = fgetc(fp);
		while (c!=']' && c != EOF){	
			if (array_size< MAX_ARRAY){
			   if (c!=' '){
				while (c!=']'&&c!=','){ 
					if (c==EOF){
						printf("Premature end of file\n");
						fclose(fp);
						exit(-1);
					}
					if (index >= MAX_VECTOR_STR-1){
						printf("Number is surpassing the maximum number of characters\n");
						fclose(fp);
						exit(-1);
					}
					str1[index]=c;
					index++;
					c = fgetc(fp);
					//printf("character: %c",c);
				}
				str1[index]='\0';
				index=0;
			 	//printf("String loaded: %s\n", str1); 	
				array[array_size]=atoi(str1);
				array_size++;
			  }
 			  c = fgetc(fp);
	 	          //printf("character: %c",c);

			}else{
				printf("Number of elements of the array exceeds the maximum\n");
				fclose(fp);
                                exit(-1);
			} 	
		}
	}else{
		printf("Format do not match with the expected array template\n");
		fclose(fp);
                exit(-1);
	}
   	fclose(fp);
   }else{
	printf("Error writing file %s\n",fileName);
        exit(-1);
   }

   printf("Array size is: %d\n[ ", array_size);
   int i;
   for(i=0; i<array_size; i++)
   	printf("%d ",array[i]);
   printf("]\n");
   return array_size;
}

void writeArray(char* fileName, int* array, int array_size){
  char str1[MAX_VECTOR_STR], num[MAX_VECTOR_STR];
  int i;
  strcpy(str1,"[");
  for (i=0; i< array_size; i++){
	sprintf(num, "%d",array[i]);
	strcat(str1, num);
	if (i < (array_size-1)){
		strcat(str1, ", ");
	}else{
		strcat(str1, "]");
	}
		
  }
  printf("New array %s\n",str1);
  FILE * fp = fopen (fileName, "w");
  if (fp){
	fputs(str1,fp);
  	fclose(fp);
  }else{
	printf("Error writing file %s\n",fileName);
	exit(-1);
  }
}
int main (int argc, char** argv) {
  int my_rank, size;
  char* fileName;
  int* array = malloc(sizeof(int)*MAX_ARRAY);
  int array_size = 0, i=0, source=0;
  // Initialize the MPI environment
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
  MPI_Comm_rank (MPI_COMM_WORLD, &my_rank);
  
  if (my_rank == 0) {
	printf("Checking parameters\n");
  	if (argc != 2){
        	printf("Incorrect name of parameters\n");
        	return -1;
  	}else{
        	fileName = argv[1];
        	printf("File name is %s\n", fileName);
  	}
  	array_size = loadArray(fileName, array);
  }
  MPI_Bcast ( &array_size, 1, MPI_INT, 0, MPI_COMM_WORLD );

  int chunksize = ceil((((double)array_size)/ size));
  printf("Rank %d: ArraySize: %d, chucksize: %d",my_rank,array_size, chunksize);
  MPI_Bcast ( array, array_size, MPI_INT, 0, MPI_COMM_WORLD );
  for (i = my_rank*chunksize; (i < (my_rank+1)*chunksize && i< array_size); i++){
	array[i]++;
        printf("Rank %d: Array[%d]: %d",my_rank,i, array[i]);
  }
  if (my_rank != 0){
    if (my_rank*chunksize < array_size){
	printf("Rank %d: sending: %d",my_rank,array[my_rank*chunksize]);
	if ((my_rank+1)*chunksize <= array_size){
    		MPI_Send(&array[my_rank*chunksize], chunksize, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}else{
		int sentsize = array_size - my_rank*chunksize;
		MPI_Send(&array[my_rank*chunksize], sentsize, MPI_INT, 0, 0, MPI_COMM_WORLD);
	}
    }	
  } else {
    for (source = 1; source < size; source++) {
       	if (source*chunksize < array_size){
		if ((my_rank+1)*chunksize <= array_size){
      			MPI_Recv(&array[source*chunksize], chunksize, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      		}else{
			int sentsize = array_size - my_rank*chunksize;
			MPI_Recv(&array[source*chunksize], sentsize, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		printf("Rank %d: received: %d",my_rank,array[source*chunksize]);
	}
    }
  }

  if (my_rank == 0) {
    printf("Finished writting array \n");
    writeArray(fileName, array, array_size);
  }
 
  // Finalize the MPI environment
  MPI_Finalize();
  
  return 0;
}

