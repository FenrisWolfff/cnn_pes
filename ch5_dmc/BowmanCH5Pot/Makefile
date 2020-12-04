OBJ  =  checkcheck.o fullpot.o

OPT = -O2 

main :
	python3 -m numpy.f2py -c fullpot.f -m libCH5pot

makeold : $(OBJ) 
	gfortran $(OPT) $(OBJ)  -o compcalcpot

touchall :
	touch *.f

all : touchall compcalcpot main


.f.o : 
	gfortran $(OPT) -c  $<

.c.o :
	cc  $(OPT) -c  $<
