GCC=g++
MCC=mcxx
KC=nvcc
#CFLAGS=-std=c++11 -DDEBUG_BINDING -g -I. -I/opt/COMPSs/Bindings/c/include -I/opt/COMPSs/Bindings/bindings-common/include
CFLAGS=-std=c++11 -g -fpermissive -I. -O3 -Wall -Wno-unused -I/apps/BOOST/1.57/include -I/apps/COMPSs/Trunk_adria/Bindings/c/include -I/apps/COMPSs/Trunk_adria/Bindings/bindings-common/include
CFILES=Clusters.cc ClustersSize.cc Fragment.cc wtime.cc file_io.cc
CFILES2=Clusters.cc ClustersSize.cc Fragment.cc wtime.cc file_io.cc
C_OBJ_MASTER=Clusters.o ClustersSize.o Fragment.o wtime.o file_io.o
C_OBJ_WORKER=Clusters.o ClustersSize.o Fragment.o wtime.o file_io.o kernel.o


all: clean compile_master compile_worker deploy

compile_master:
	@echo "Building application for master..."
	$(MCC) -DCOMPSS_MASTER $(CFLAGS) -c $(CFILES)
	ar rvs libmaster.a $(C_OBJ_MASTER)
	ranlib libmaster.a

compile_worker: kernel.o
	@echo "Building application for workers..."
	$(MCC) --ompss -DCOMPSS_WORKER $(CFLAGS) -c $(CFILES) 
	ar rvs libworker.a $(C_OBJ_WORKER)
	ranlib libworker.a

kernel.o:
	@echo "Building kernel..."
	$(KC) -o kernel.o -c kernel.cu

deploy:
	@echo "Deploying lib files..."
	@rm -rf ../lib
	@mkdir ../lib
	@mv *.a ../lib

clean:
	@echo "Cleaning intermediate files..."
	@rm -f *.o

