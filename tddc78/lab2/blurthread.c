//standard imports
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include "ppmio.h"
#include "gaussw.h"

//defines
#define MAX_RAD 1000
#define MAX_PIXELS (1000*1000)
#define MAX_THREADS 100
#define BILLION  1000000000L;

int min(int a, int b) {
	if (a < b) {
		return a;
	} else {
		return b;
	}
}

struct data_to_thread {
	unsigned char * src;
	unsigned char * dst;
	int xsize, ysize, start_y, end_y, radius;

} data_to_thread;

static void * blurfiltermpi(void *arguments) {
	struct data_to_thread* data = (struct data_to_thread *) arguments;
	unsigned char *src = data->src;
	unsigned char *dst = data->dst;
	int xsize = data->xsize;
	int ysize = data->ysize;
	int start_y = data->start_y;
	int end_y = data->end_y;
	int radius = data->radius;
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
			for (int sub_y = y - radius; sub_y < y + radius; sub_y++) {
				if (sub_y < 0 || sub_y > ysize) {
					continue;

				}
				//for each sub column
				for (int sub_x = x - radius; sub_x < x + radius; sub_x++) {
					if (sub_x < 0 || sub_x > xsize) {
						continue;
					}
					temp_weight = w[min(abs(sub_x - x), abs(sub_y - y))];
					temp_pointer = src + (xsize * sub_y + sub_x) * 3;
					red += temp_weight * (*temp_pointer);
					green += temp_weight * (*(temp_pointer + 1));
					blue += temp_weight * (*(temp_pointer + 2));
					n += temp_weight;
				}

			}
			temp_pointer = dst + (xsize * y + x) * 3;
			*temp_pointer = red / n;
			*(temp_pointer + 1) = green / n;
			*(temp_pointer + 2) = blue / n;

		}

	}
	return 0;
}

int main(int argc, char ** argv) {
	//used to know how blurry the image will be
	int radius;

	// information about the picture
	int xsize, ysize, colmax, number_of_threads;

	char* src = (char*) malloc(sizeof(unsigned char) * MAX_PIXELS * 3);
	char* dst = (char*) malloc(sizeof(unsigned char) * MAX_PIXELS * 3);

	struct timespec start, stop;
	double accum;

	if (argc != 5) {
		fprintf(stderr, "Usage: %s radius threads infile outfile\n", argv[0]);
		exit(1);
	}
	radius = atoi(argv[1]);
	number_of_threads = atoi(argv[2]);
	if ((radius > MAX_RAD) || (radius < 1)) {
		fprintf(stderr,
				"Radius (%d) must be greater than zero and less then %d\n",
				radius, MAX_RAD);
		exit(1);
	}
	/* read file */
	if (read_ppm(argv[3], &xsize, &ysize, &colmax, (char *) src) != 0) {
		exit(1);
	}
	if (colmax > 255) {
		fprintf(stderr, "Too large maximum color-component value\n");
		exit(1);
	}

	int number_of_elements, temp_start, temp_end;

	struct data_to_thread* arguments[MAX_THREADS];
	pthread_t* threads[MAX_THREADS];
	struct data_to_thread* temp_arguments;
	 clock_gettime( CLOCK_REALTIME, &start);
	for (unsigned int i = 1; i < number_of_threads; i++) {
		temp_start = i * ysize / number_of_threads;
		if (temp_start < 0) {
			temp_start = 0;
		}
		temp_end = (i + 1) * ysize / number_of_threads;
		if (temp_end > ysize) {
			temp_end = ysize;

		}
		if (i == number_of_threads - 1) {
			temp_end = ysize;
		}
		temp_arguments = (struct data_to_thread*) malloc(
				sizeof(struct data_to_thread));
		temp_arguments->end_y = temp_end;
		temp_arguments->radius = radius;
		temp_arguments->src = src;
		temp_arguments->start_y = temp_start;
		temp_arguments->xsize = xsize;
		temp_arguments->ysize = ysize;
		temp_arguments->dst = dst;
		arguments[i] = temp_arguments;
		threads[i] = (pthread_t*) malloc(sizeof(pthread_t));
		pthread_create(threads[i], NULL, &blurfiltermpi, (void *) arguments[i]);
	}
	temp_arguments = (struct data_to_thread*) malloc(
			sizeof(struct data_to_thread));

	temp_end = (1) * ysize / number_of_threads;
	if (temp_end > ysize) {
		temp_end = ysize;

	}
	if (0 == number_of_threads - 1) {
		temp_end = ysize;
	}
	temp_arguments->end_y = temp_end;
	temp_arguments->radius = radius;
	temp_arguments->src = src;
	temp_arguments->start_y = 0;
	temp_arguments->xsize = xsize;
	temp_arguments->ysize = ysize;
	temp_arguments->dst = dst;
	arguments[0] = temp_arguments;
	blurfiltermpi((void *) temp_arguments);
	 clock_gettime( CLOCK_REALTIME, &stop);
	for (unsigned int i = 1; i < number_of_threads; i++) {
		pthread_join(*threads[i], NULL);
	}
    accum = ( (double)stop.tv_sec - (double)start.tv_sec )
          + ( (double)stop.tv_nsec - (double)start.tv_nsec )
            / BILLION;
    printf( "it took: %lf\n", accum );
	write_ppm(argv[4], xsize, ysize, dst);
	free(dst);
	free(src);
	for (int i = 0; i < number_of_threads; i++) {
		free(arguments[i]);
		if (i > 0) {
			free(threads[i]);
		}
	}

	return 0;
}
