//standard imports
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>


//non-standard imports
#include "ppmio.h"
#include "blurfilter.h"
#include "gaussw.h"


//defines
#define MAX_PIXELS (1000*1000)


unsigned int calculcatethres(unsigned char *src, int xsize, int ysize, int start_y,
		int end_y) {
	unsigned char* temp_pointer;
	unsigned int sum=0;
	//for each row
	for (int y = start_y; y < end_y; y++) {
		//for each column
		for (int x = 0; x < xsize; x++) {
			temp_pointer = src + (xsize * y + x) * 3;
			sum+=*temp_pointer+*(temp_pointer+1)+*(temp_pointer+2);
		}

	}
	return sum;

}

void thresfiltermpi(unsigned char *src, int xsize, int ysize, int start_y,
		int end_y, unsigned thres) {
	unsigned char* temp_pointer;
	for (int y = start_y; y < end_y; y++) {
		//for each column
		for (int x = 0; x < xsize; x++) {
			 temp_pointer = src + (xsize * y + x) * 3;
			 if (*temp_pointer+*(temp_pointer+1)+*(temp_pointer+2)>thres){
					*temp_pointer = 255;
					*(temp_pointer + 1) = 255;
					*(temp_pointer + 2) = 255;
			 }
			 else{
			*temp_pointer = 0;
			*(temp_pointer + 1) = 2;
			*(temp_pointer + 2) = 0;

			 }
			 }

	}

}

int main(int argc, char ** argv) {
	//used to know where we are in the hierarchy
	int rank, size;

	//used to know how blurry the image will be
	int radius;

	 double starttime, endtime;

	// information about the picture
	int xsize, ysize, colmax;

	//store picture
	pixel src[MAX_PIXELS];

	MPI_Init(&argc, &argv); /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */

	if (argc != 3) {
		fprintf(stderr, "Usage: %s infile outfile\n", argv[0]);
		exit(1);
	}
	radius = atoi(argv[1]);

	if (rank == 0) {
		/* read file */
		if (read_ppm(argv[1], &xsize, &ysize, &colmax, (char *) src) != 0) {
			exit(1);
		}
		if (colmax > 255) {
			fprintf(stderr, "Too large maximum color-component value\n");
			exit(1);
		}

	}
	starttime = MPI_Wtime();

	//send problem size to slaves
	MPI_Bcast(&xsize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&ysize, 1, MPI_INT, 0, MPI_COMM_WORLD);
	int start_y = rank * ysize / size;
	int end_y = (rank + 1) * ysize / size;
	if (rank == size - 1) {
		end_y = ysize;
	}
	//send data so slaves
	int temp_start;
	int temp_end;
	if (rank == 0) {
		int number_of_elements;
		for (unsigned int i = 1; i < size; i++) {
			int temp_start = i * ysize / size - radius;
			if (temp_start < 0) {
				temp_start = 0;
			}
			int temp_end = (i + 1) * ysize / size + radius;
			if (temp_end > ysize) {
				temp_end = ysize;

			}
			if (i == size - 1) {
				temp_end = ysize;
			}
			number_of_elements = (temp_end - temp_start) * 3 * xsize;
			temp_start = temp_start * 3 * xsize;

			MPI_Request temp_request;
			MPI_Isend((char *) src + temp_start, number_of_elements, MPI_CHAR,
					i, 0, MPI_COMM_WORLD, &temp_request);
		}
	} else {

		temp_start = rank * ysize / size - radius;
		if (temp_start < 0) {
			temp_start = 0;
		}
		temp_end = (rank + 1) * ysize / size + radius;
		if (temp_end > ysize) {
			temp_end = ysize;

		}
		if (rank == size - 1) {
			temp_end = ysize;
		}

		int number_of_elements = (temp_end - temp_start) * 3 * xsize;
		int start = temp_start * 3 * xsize;
		MPI_Status temp_status;
		MPI_Recv((char *) src + start, number_of_elements, MPI_CHAR, 0, 0,
				MPI_COMM_WORLD, &temp_status);

	}

	//do works

	temp_start = rank * ysize / size;
	if (temp_start < 0) {
		temp_start = 0;
	}
	temp_end = (rank + 1) * ysize / size;
	if (temp_end > ysize) {
		temp_end = ysize;

	}
	if (rank == size - 1) {
		temp_end = ysize;
	}
	unsigned int thres=calculcatethres((unsigned char *) src, xsize, ysize, temp_start, temp_end);

	if (rank==0){
		unsigned int temp=0;
		for (unsigned int i = 1; i < size; i++) {

			MPI_Request temp_request;
			MPI_Status temp_status1;
			MPI_Recv(&temp, 1, MPI_UNSIGNED, i,
					0, MPI_COMM_WORLD, &temp_status1);
			thres+=temp;
		}
		thres=thres/(xsize*ysize);

	}
	else{
		MPI_Send(&thres, 1, MPI_UNSIGNED, 0, 0,
				MPI_COMM_WORLD);

	}
	MPI_Bcast(&thres, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

	thresfiltermpi((unsigned char *)src, xsize,ysize,temp_start,temp_end,thres);

	if (rank == 0) {
		int temp_start;
		int temp_end;
		int number_of_elements;
		pixel dst[MAX_PIXELS];
		for (unsigned int i = 1; i < size; i++) {
			int temp_start = i * ysize / size;
			int temp_end = (i + 1) * ysize / size;
			if (i == size - 1) {
				temp_end = ysize;
			}
			number_of_elements = (temp_end - temp_start) * 3 * xsize;
			temp_start = temp_start * 3 * xsize;

			MPI_Request temp_request;
			MPI_Status temp_status2;
			MPI_Recv((char *) dst + temp_start, number_of_elements, MPI_CHAR, i,
					0, MPI_COMM_WORLD, &temp_status2);
		}
		memcpy(dst, src, sizeof(char) * end_y * xsize * 3);
		write_ppm(argv[2], xsize, ysize, (char *) dst);
	} else {
		int number_of_elements = (end_y - start_y) * 3 * xsize;
		int start = start_y * 3 * xsize;
		MPI_Send((char *) src + start, number_of_elements, MPI_CHAR, 0, 0,
				MPI_COMM_WORLD);

	}
	endtime = MPI_Wtime();
	if (rank==0){
		printf("That took %f seconds\n",endtime-starttime);
	}

	MPI_Finalize();
}
