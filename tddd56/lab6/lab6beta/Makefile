CC      = gcc
SHARED	=		0
CFLAGS  = -I $(shell pwd)  -lOpenCL -lglut -lGL -I/usr/local/cuda/include -D SHARED=$(SHARED) -O3
LDFLAGS = -I $(shell pwd)  -lOpenCL -lglut -lGL -I/usr/local/cuda/include -D SHARED=$(SHARED) -O3


all: filter
.PHONY : all


filter:  filter.o CLutilities.o milli.o readppm.o
	$(CC) -o $@ $^ $(LDFLAGS) 
	rm -f *.o *~>//dev/null
	
filter.o: filter.c
	$(CC) -c $(CFLAGS) $<

milli.o: milli.c milli.h
	$(CC) -c $(CFLAGS) $<

CLutilities.o: CLutilities.c CLutilities.h
	$(CC) -c $(CFLAGS) $<

readppm.o: readppm.c readppm.h
	$(CC) -c $(CFLAGS) $<


.PHONY: clean cleanest

clean:
	rm -f filter *.o *~>//dev/null
