CC=icc
MCC=icc
CFLAGS = 


LFLAGS= -std=c99 -lpthread -lrt -lm -g -tcollect -I$(VT_ROOT)/include -Nmpi
all: blurmpi


clean:
	-$(RM) *.o blur thresc blurmpi


blurc: ppmio.o gaussw.o blurfilter.o blurmain.o
	$(CC) -o $@ ppmio.o gaussw.o blurfilter.o blurmain.o $(LFLAGS) 

thresc: thresmain.o ppmio.o thresfilter.o
	$(CC)  -o $@ thresmain.o ppmio.o thresfilter.o $(LFLAGS)

arc:
	tar cf - *.c *.h *.f90 Makefile Makefile.sgi|gzip - > filters.tar.gz
	
blurmpi : ppmio.o gaussw.o blurfilter.o blurmainmpi.o
	$(MCC) -o $@ ppmio.o gaussw.o blurfilter.o blurmainmpi.o $(LFLAGS)
	
blurmainmpi.o :
	$(MCC) -c -o blurmainmpi.o blurmainmpi.c $(LFLAGS)
	
blurmain.o : 
	$(CC) -c -o blurmain.o blurmain.c $(LFLAGS) 
	

thresmpi : ppmio.o gaussw.o blurfilter.o thresfiltermpi.o
	$(MCC) -o $@ ppmio.o gaussw.o blurfilter.o thresfiltermpi.o $(LFLAGS)
	
thresfiltermpi.o :
	$(MCC) -c -o thresfiltermpi.o thresfiltermpi.c $(LFLAGS)
