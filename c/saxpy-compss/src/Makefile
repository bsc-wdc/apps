GCC=g++
MCC=mcxx
CFLAGS=-std=gnu++11 -fpermissive -DDEBUG_BINDING -g -I. -I/opt/COMPSs/Bindings/c/include -I/opt/COMPSs/Bindings/bindings-common/include

EXTRA = -O3 -Wall -Wno-unused -g --variable=disable_opencl_file_check:1 -march=native -mavx -msse2 -ffast-math

CFILES=saxpy_obj.cc
C_OBJ=saxpy_obj.o


all: clean compile_master compile_worker deploy

compile_master:
	@echo "Building application for master..."
	$(MCC) -DCOMPSS_MASTER $(CFLAGS) $(EXTRA) $(INCS) -c $(CFILES) $(LIBS)
	ar rvs libmaster.a $(C_OBJ)
	ranlib libmaster.a

compile_worker:
	@echo "Building application for workers..."
	$(MCC) --ompss --variable=nanos-debug:1 -DCOMPSS_WORKER $(CFLAGS) $(EXTRA) $(INCS) -c $(CFILES) $(LIBS)
	ar rvs libworker.a $(C_OBJ)
	ranlib libworker.a

deploy:
	@echo "Deploying lib files..."
	@rm -rf ../lib
	@mkdir ../lib
	@mv *.a ../lib

clean:
	@echo "Cleaning intermediate files..."
	@rm -f *.o
