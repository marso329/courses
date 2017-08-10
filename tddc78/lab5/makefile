CC=g++

LFLAGS= -std=c++11 -O3 -Nmpi -Wall

all: main

.PHONY : all


main:
	module add gcc/5.3.0
	$(CC) -o main main.cpp $(LFLAGS)
	

.PHONY : clean

clean :
	rm -f main>//dev/null	
