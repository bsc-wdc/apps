GCC=g++
MCC=mcxx
#MCC=mnvcxx
KC=nvcc
#CFLAGS=-std=c++11 -DDEBUG_BINDING -g -I. -I/opt/COMPSs/Bindings/c/include -I/opt/COMPSs/Bindings/bindings-common/include
CFLAGS=-std=c++11 -g -I. -I/apps/COMPSs/Trunk_adria/Bindings/c/include -I/apps/COMPSs/Trunk_adria/Bindings/bindings-common/include -I$(BOOST_HOME)/include -O3 -Wall -Wno-unused
#-I/opt/COMPSs/Bindings/c/include -I/opt/COMPSs/Bindings/bindings-common/include
CFLAGS2=-std=c++11 -g -I. -I/apps/COMPSs/Trunk/Bindings/c/include -I/apps/COMPSs/Trunk_adria/Bindings/bindings-common/include -I$(BOOST_HOME)/include -K


CFILES=Block.cc Matrix.cc
C_OBJ_MASTER=Block.o Matrix.o 
C_OBJ_WORKER=Block.o Matrix.o kernel.o


all: clean compile_master compile_worker deploy

compile_master: 
	@echo "Building application for master..."
	$(GCC) -DCOMPSS_MASTER $(CFLAGS) -c $(CFILES) 
	ar rvs libmaster.a $(C_OBJ_MASTER)
	ranlib libmaster.a

compile_worker: kernel.o
	@echo "Building application for workers..."
	$(MCC) --ompss -DCOMPSS_WORKER $(CFLAGS2) -c Block.cc -o Block.o 
	$(MCC) --ompss -DCOMPSS_WORKER $(CFLAGS2) -c Matrix.cc -o Matrix.o
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

