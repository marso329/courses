#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "ppmio.h"
#include "blurfilter.h"
#include "gaussw.h"
#define MAX_RAD 1000

int counter = 0;


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
	int radius;
	int xsize, ysize, colmax;
	pixel src[MAX_PIXELS];
#define MAX_RAD 1000

	double w[MAX_RAD];

	/* Take care of the arguments */

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

	/* read file */
	if (read_ppm(argv[2], &xsize, &ysize, &colmax, (char *) src) != 0)
		exit(1);

	if (colmax > 255) {
		fprintf(stderr, "Too large maximum color-component value\n");
		exit(1);
	}

	printf("Has read the image, generating coefficients\n");

	/* filter */
	get_gauss_weights(radius, w);

	printf("Calling filter\n");

	// blurfilter(xsize, ysize, src, radius, w);
	blurfiltermpi((unsigned char*) src, xsize, ysize, 0, ysize, radius);

	/* write result */
	printf("Writing output file\n");
	printf("counter %i \n",counter);

	if (write_ppm(argv[3], xsize, ysize, (char *) src) != 0)
		exit(1);

	return (0);
}
