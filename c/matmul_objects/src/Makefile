all:

	@echo "Building application..."

	g++ -DCOMPSS_MASTER -g -I. -I/opt/COMPSs/Runtime/bindings/c/include -I/opt/COMPSs/Runtime/bindings/bindings-common/include -c Block.cc Matrix.cc
		
	ar rvs libmaster.a Block.o Matrix.o
	ranlib libmaster.a
	
	rm *.o
		
	g++ -DCOMPSS_WORKER -g -I. -I/opt/COMPSs/Runtime/bindings/c/include -I/opt/COMPSs/Runtime/bindings/bindings-common/include -c Block.cc Matrix.cc
	
	ar rvs libworker.a Block.o Matrix.o
	ranlib libworker.a 
	
	rm *.o		
	 			
	@rm -rf ../lib
	@mkdir ../lib
	@mv *.a ../lib
	
	@echo " " 