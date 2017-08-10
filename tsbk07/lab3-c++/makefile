LDIR=lib
SDIR=src
CC=g++
CFLAGS=-I$(IDIR) -I$(SDIR) -std=c++11 -Wall -g
LIBS=-lglut -lGL -lGLU -lGLEW -lX11 -lm -lXrandr -lXi -lXxf86vm -lpthread -ldl
DEPS=lib/GL_utilities.c lib/VectorUtils3.c lib/loadobj.c lib/LoadTGA.c lib/MicroGlut.c

all: main
.PHONY : all

main:
	$(CC) $(CFLAGS) $(DEPS) -o main $(SDIR)/main.cpp $(LIBS)



.PHONY: clean
clean :
	rm -f main *.o *~>//dev/null



