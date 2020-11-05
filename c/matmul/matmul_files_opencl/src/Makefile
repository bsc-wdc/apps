GCC=g++
MCC=mcxx
OCC=oclmcxx
KC=nvcc
#CFLAGS=-std=c++0x -g -fpermissive -I. -I/apps/COMPSs/Trunk2/Bindings/c/include -I/apps/COMPSs/Trunk2/Bindings/bindings-common/include -I/apps/BOOST/1.57/include -O3 -Wall -Wno-unused
#-I/opt/COMPSs/Bindings/c/include -I/opt/COMPSs/Bindings/bindings-common/include
#CFLAGS2=-std=c++11 -g -I. -I/apps/COMPSs/Trunk2/Bindings/c/include -I/apps/COMPSs/Trunk2/Bindings/bindings-common/include -I/apps/BOOST/1.57/include -K
CFLAGS=-std=c++11 -g -I. -I/opt/COMPSs/Bindings/c/include -I/opt/COMPSs/Bindings/bindings-common/include -O3 -Wall -Wno-used
CFILES=Block.cc Matrix.cc
C_OBJ_MASTER=Block.o Matrix.o 
#C_OBJ_WORKER=Block.o Matrix.o kernel.o
C_OBJ_WORKER=Block.o Matrix.o


all: clean compile_master compile_worker deploy

compile_master: 
	@echo "Building application for master..."
	$(GCC) -DCOMPSS_MASTER $(CFLAGS) -c $(CFILES) 
	ar rvs libmaster.a $(C_OBJ_MASTER)
	ranlib libmaster.a

compile_worker:
	@echo "Building application for workers..."
	$(OCC) --ompss -k -DCOMPSS_WORKER $(CFLAGS) -c Block.cc /home/aaguila/subversion/c/matmul_files_opencl/src/kernel.cl --opencl-build-opts=-I/home/aaguila/subversion/c/matmul_files_opencl/src/ -o Block.o 
	$(OCC) --ompss -k -DCOMPSS_WORKER $(CFLAGS) -c Matrix.cc -o Matrix.o
	ar rvs libworker.a $(C_OBJ_WORKER)
	ranlib libworker.a

#kernel.o: 
#	@echo "Building kernel..."
#	$(MCC) -o kernel.o -c kernel-old.cu


deploy:
	@echo "Deploying lib files..."
	@rm -rf ../lib
	@mkdir ../lib
	@mv *.a ../lib

clean:
	@echo "Cleaning intermediate files..."
	@rm -f *.o

