CC=gcc

CFLAGS = 

LFLAGS= -std=c11 -D_XOPEN_SOURCE=600 -lpthread -lrt -lm -g -O3
all: blurthread thresthread

clean:
	-$(RM) *.o blurthread threthread

blurthread: ppmio.o gaussw.o blurthread.o
	$(CC) -o $@ ppmio.o gaussw.o blurthread.o $(LFLAGS) 

gaussw.o : 
	$(CC) -c -o gaussw.o gaussw.c $(LFLAGS) 
	
ppmio.o : 
	$(CC) -c -o ppmio.o ppmio.c
	
blurthread.o : 
	$(CC) -c -o blurthread.o blurthread.c $(LFLAGS) 
	
	
thresthread: ppmio.o gaussw.o thresthread.o 
	$(CC) -o $@ ppmio.o gaussw.o thresthread.o $(LFLAGS)
	
thresthread.o : 
	$(CC) -c -o thresthread.o thresthread.c $(LFLAGS) 

	
