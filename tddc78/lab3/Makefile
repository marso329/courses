CC=icc
THREADS=1
CFLAGS= -std=c99 -Wall -O3 -openmp 

all:laplsolv lapsolvomp

clean:
	rm laplsolv lapsolvomp



laplsolv:laplsolv.f90
	gfortran -fopenmp $^ -o $@

lapsolvomp:
	$(CC) $(CFLAGS) lapsolvomp.c -o $@
