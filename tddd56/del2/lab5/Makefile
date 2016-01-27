CC      = c++
NVCC	= nvcc
CFLAGS  = -I $(shell pwd) -std=c++0x -pthread -g 
LDFLAGS = -std=c++0x -pthread -g
NCFLAGS = -x cu -arch=sm_20
NLDFLAGS= -arch=sm_20
NDEF	= GPU
K			= 	16
PRINT	=		32

all: cpu gpu
.PHONY : all

cpu:  bitonic_main.o milli.o
	$(CC) -o $@ $^ $(LDFLAGS) -D K=$K -D PRINT=$(PRINT)
	rm -f *.o *~>//dev/null
	
bitonic_main.o: bitonic_main.cpp
	export MAKEFLAGS="-j $(grep -c ^processor /proc/cpuinfo)"
	$(CC) -c $(CFLAGS) $< -D K=$K -D PRINT=$(PRINT)

milli.o: milli.c milli.h
	$(CC) -c $(CFLAGS) $< -D K=$K -D PRINT=$(PRINT)

bitonic_kernel.o: bitonic_kernel.cu bitonic_kernel.h 
	$(NVCC)  $(NCFLAGS) -I. -dc $< -o $@ -D $(NDEF) -D K=$K -D PRINT=$(PRINT)

bitonic_main_gpu.o: bitonic_main.cpp
	export MAKEFLAGS="-j $(grep -c ^processor /proc/cpuinfo)"
	$(NVCC)  -$(NCFLAGS) -I. -dc $< -o $@  -D $(NDEF) -D K=$K -D PRINT=$(PRINT)
 
milli_gpu.o: milli.c milli.h
	$(NVCC)  $(NCFLAGS) -I. -dc $< -o $@  -D $(NDEF) -D K=$K -D PRINT=$(PRINT)

gpu: bitonic_main_gpu.o milli_gpu.o bitonic_kernel.o
	$(NVCC) $(NLDFLAGS) -o $@ $^  -D $(NDEF) -D K=$K -D PRINT=$(PRINT)
	rm -f *.o *~>//dev/null


.PHONY: clean cleanest

clean:
	rm -f cpu gpu *.o *~>//dev/null
