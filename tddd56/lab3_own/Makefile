CC      = g++
CFLAGS  = -I $(shell pwd) -std=c++0x -pthread -g
LDFLAGS = -std=c++0x -pthread -g

all: sort clean

sort: sort.o
	$(CC) -o $@ $^ $(LDFLAGS)

sort.o: sort.cpp sort.h
	$(CC) -c $(CFLAGS) $<

.PHONY: clean cleanest

clean:
	rm *.o

cleanest: clean
	rm sort
