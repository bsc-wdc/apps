DEBUG =
#DEBUG = -g -fbacktrace -fbounds-check 
#DEBUG = $(DEBUG) -DDEBUG
PROFILE =
#PROFILE = -pg
PARALLEL =
# OMP
#PARALLEL =  -fopenmp -DOMP
#
#OPTIONS = -O3 $(PARALLEL) -fimplicit-none -funroll-loops -mtune=native -march=native -Wall -ftree-vectorizer-verbose=1 -ffast-math  
#OPTIONS = -O3 $(PARALLEL) -fimplicit-none -funroll-loops -mtune=generic  -Wall -ftree-vectorizer-verbose=1 -ffast-math  
OPTIONS = -O3 $(PARALLEL) -fimplicit-none -funroll-loops -Wall -ftree-vectorizer-verbose=1 -ffast-math
#
LOPTIONS=   $(PARALLEL) 
OBJS = IO.o random.o energy.o colisio.o utils.o
MODS = commLine.o geometry.o geometryDP.o stepPotentials.o intList.o paramSet.o
INCS = vars.h
F77= gfortran
#F77= /gpfs/apps/GCC/4.6.1/bin/gfortran
CPP= cpp
CFLAGS = -c $(DEBUG) $(PROFILE) 
LFLAGS = $(PROFILE)

discrete: ${MODS} $(INCS) $(OBJS) main.o
	$(F77) $(LFLAGS) -o ../exe/discrete0.2.4 $(LOPTIONS) $(OBJS) $(MODS) main.o
.f.o: $(INCS) $<
	$(F77) -ffree-form $(OPTIONS) $(PC32) $(CFLAGS) $<
.F.o: $(INCS) $<
	$(F77) -ffree-form $(OPTIONS) $(PC32) $(CFLAGS) $<
clean: 
	rm *.o *.mod ../exe/discrete0.2.4
