//standard imports
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <mpi.h>
#include <VT.h>

//non-standard imports
#include "ppmio.h"
#include "blurfilter.h"
#include "gaussw.h"

//defines
#define MAX_RAD 1000

int min(int a,int b){
	if (a<b){
		return a;
	}
	else{
		return b;
	}
}

void blurfiltermpi(unsigned char *src, int xsize, int ysize, int start_y,
		int end_y, int radius) {
	double w[MAX_RAD];
	get_gauss_weights(radius, w);
	double red, green, blue, n, temp_weight;
	register unsigned char* temp_pointer;
	//for each row
	for (int y = start_y; y < end_y; y++) {
		//for each column
		for (int x = 0; x < xsize; x++) {
			red = 0.0;
			green = 0.0;
			blue = 0.0;
			n = 0.0;
			//for each sub row
			for (int sub_y=y-radius;sub_y<y+radius;sub_y++){
				if (sub_y<0 || sub_y>ysize){
					continue;

				}
				//for each sub column
				for(int sub_x=x-radius;sub_x<x+radius;sub_x++){
					if (sub_x<0 || sub_x>xsize){
						continue;
					}
					temp_weight=w[min(abs(sub_x-x),abs(sub_y-y))];
					temp_pointer = src + (xsize * sub_y + sub_x) * 3;
					red += temp_weight * (*temp_pointer);
					green += temp_weight * (*(temp_pointer + 1));
					blue += temp_weight * (*(temp_pointer + 2));
					n += temp_weight;
				}

			}
			 temp_pointer = src + (xsize * y + x) * 3;
			*temp_pointer = red / n;
			*(temp_pointer + 1) = green / n;
			*(temp_pointer + 2) = blue / n;

		}

	}

}

int main(int argc, char ** argv) {
	VT_initialize(&argc, &argv);
	int func_statehandle;
	VT_funcdef("Main", VT_NOCLASS, &func_statehandle);
VT_enter(func_statehandle, VT_NOSCL);	
//used to know where we are in the hierarchy
	int rank, size;

	//used to calc time
	double starttime, endtime;

	//used to know how blurry the image will be
	int radius;

	// information about the picture
	int xsize, ysize, colmax;

	//store picture
	pixel src[MAX_PIXELS];

	MPI_Init(&argc, &argv); /* starts MPI */
	MPI_Comm_rank(MPI_COMM_WORLD, &rank); /* get current process id */
	MPI_Comm_size(MPI_COMM_WORLD, &size); /* get number of processes */
//	int *temp_int=0;
//	int stupid=*temp_int;
	if (argc != 4) {
		fprintf(stderr, "Usage: %s radius infile outfile\n", argv[0]);
		exit(1);
	}
	radius = atoi(argv[1]);
	if ((radius > MAX_RAD) || (radius < 1)) {
		fprintf(stderr,
				"Radius (%d) must be greater than zero and less then %d\n",
				radius, MAX_RAD);
		exit(1);
	}

	if (rank == 0) {
		/* read file */
		if (read_ppm(argv[2], &xsize, &ysize, &colmax, (char *) src) != 0) {
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
//	int* temp_int=0;
//	int stupid=*temp_int;
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
		MPI_Status temp_status1;
		MPI_Recv((char *) src + start, number_of_elements, MPI_CHAR, 0, 0,
				MPI_COMM_WORLD, &temp_status1);

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
	int blur_symdef;
		VT_funcdef("Filter:Blur", VT_NOCLASS, &blur_symdef);
		VT_enter(blur_symdef, VT_NOSCL);

	blurfiltermpi((unsigned char *) src, xsize, ysize, temp_start, temp_end,
			radius);
VT_leave(VT_NOSCL);
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
			MPI_Status temp_status;
			MPI_Recv((char *) dst + temp_start, number_of_elements, MPI_CHAR, i,
					0, MPI_COMM_WORLD, &temp_status);
		}
		memcpy(dst, src, sizeof(char) * end_y * xsize * 3);
		write_ppm(argv[3], xsize, ysize, (char *) dst);
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
VT_leave(VT_NOSCL);
	VT_finalize();
}
