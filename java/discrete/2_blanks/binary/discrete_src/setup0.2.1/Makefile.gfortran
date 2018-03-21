DEBUG =
#DEBUG = -g -fbacktrace 
#BOUNDS = -fbounds-check
PROFILE =
#PROFILE = -pg
PARALLEL =
# OMP
#PARALLEL = -openmp
#
# Binary format
#
#OPTIONS = -O3 -ffree-line-length-132  $(PARALLEL) -fimplicit-none -Wall -mtune=generic -ffast-math
OPTIONS = -O3 -ffree-line-length-132  $(PARALLEL) -fimplicit-none -Wall -ffast-math

#
LOPTIONS=   $(PARALLEL) 
OBJS = 
MODS =  utils.o commLine.o geometry.o resLibrary.o potentials.o Structure.o SetupGlobals.o
INCS = 
F77= gfortran
CPP= cpp
CFLAGS = -c $(DEBUG) $(PROFILE) $(BOUNDS)
LFLAGS = $(PROFILE)

setup: $(MODS) main.o
	$(F77) $(LFLAGS) -o ../exe/DMDSetup0.2.1 $(LOPTIONS) $(MODS) $(OBJS) $(OBJSMPI) main.o

ligandSetup: $(MODS) ligandSetup.o
	$(F77) $(LFLAGS) -o ../exe/ligandSetup $(LOPTIONS) $(MODS) $(OBJS) $(OBJSMPI) ligandSetup.o

.f.o: $(INCS) $<
	$(F77) -ffree-form $(OPTIONS) $(PC32) $(CFLAGS) $<
clean: 
	rm *.o *.mod ../exe/DMDSetup0.2.1 ../exe/ligandSetup
