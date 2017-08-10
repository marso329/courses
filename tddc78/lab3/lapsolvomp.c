#include "omp.h"
#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "math.h"
#include "semaphore.h"

#define CEILING_POS(X) ((X-(int)(X)) > 0 ? (int)(X+1) : (int)(X))
#define CEILING_NEG(X) ((X-(int)(X)) < 0 ? (int)(X-1) : (int)(X))
#define CEILING(X) ( ((X) > 0) ? CEILING_POS(X) : CEILING_NEG(X) )

int n = 1000;
int maxiter = 10000;
double error = 0.0;
double tol = 0.001;
double start_time,end_time;
double** init_t() {
	double** T = (double**) malloc(sizeof(double*) * (n + 2));
	for (int i = 0; i <= n + 1; i++) {
		T[i] = (double*) calloc(n + 2, sizeof(double));
	}
	for (int i = 0; i <= n + 1; i++) {
		if (i != n + 1) {
			T[i][0] = 1.0;
			T[i][n + 1] = 1.0;
		}
		T[n + 1][i] = 2.0;
	}

	return T;
}

void free_t(double** T) {
	for (int i = 0; i <= n + 1; i++) {
		free(T[i]);
	}
	free(T);
}

void print_t(double** T) {
	for (int i = 0; i <= n + 1; i++) {
		printf("\n");
		for (int j = 0; j <= n + 1; j++) {
			printf(" %lf ", T[i][j]);
		}
	}
	printf("\n");

}

int main(int argc, char ** argv) {
	omp_lock_t writelock;

	omp_init_lock(&writelock);
	double** T = init_t();

	double* temp;
	double* copy_before = (double*) malloc(sizeof(double) * (n + 2));
	double* copy_after = (double*) malloc(sizeof(double) * (n + 2));
	//for each computing part(couple of rows)
	int start = 0;
	int end = 0;
	//omp_set_num_threads(1);
	int chunks;
start_time=omp_get_wtime();
#pragma omp parallel shared(T,start,end,n,copy_before,copy_after,writelock,error,maxiter,tol,temp,chunks)
	{
		for (int iteration = 1; iteration < maxiter; iteration++) {
#pragma omp barrier
			memcpy(copy_before, T[0], sizeof(double) * (n + 2));
			error = 0.0;
			chunks = (int) CEILING(
					(double ) n / (double ) omp_get_num_threads());
			if (chunks <= 0) {
				chunks = 1;
			}
			int element_per_chunk = n / chunks;
			for (int i = 0; i < chunks; i++) {
				start = element_per_chunk * i + 1;
				end = element_per_chunk * (i + 1);
				if (i == chunks - 1) {
					end = n;
				}
				if (omp_get_thread_num() == 0) {
					memcpy(copy_after, T[element_per_chunk * (i + 1)],
							sizeof(double) * (n + 2));
				}
#pragma omp barrier
				int j = start + omp_get_thread_num();
				double temp_error = 0.0;
				double copy_during[n + 2];
				if (j <= end) {
					memcpy(copy_during, T[j], sizeof(double) * (n + 2));
					// for each column
					for (int k = 1; k <= n; k++) {
						if (j == start) {
							copy_during[k] = (T[j][k + 1] + T[j][k - 1]
									+ T[j + 1][k] + copy_before[k]) / 4.0;
						} else {
							copy_during[k] = (T[j][k + 1] + T[j][k - 1]
									+ T[j + 1][k] + T[j - 1][k]) / 4.0;
						}
						if (fabs(copy_during[k] - T[j][k]) > temp_error) {
							temp_error = fabs(copy_during[k] - T[j][k]);
						}
					}
				}
#pragma omp barrier
				if (j <= end) {
					omp_set_lock(&writelock);
					if (temp_error > error) {
						error = temp_error;
					}
					memcpy(T[j], copy_during, sizeof(double) * (n + 2));
					omp_unset_lock(&writelock);
					//switch places on temp storage
				}
				if (omp_get_thread_num() == 0) {
					temp = copy_after;
					copy_after = copy_before;
					copy_before = temp;
				}
			}
#pragma omp barrier
			if (error < tol) {
				if (omp_get_thread_num() == 0) {
					printf("iterations: %i\n", iteration);
				}
				break;
			}

		}
	}
	end_time=omp_get_wtime();
	printf("omp time: %lf \n",end_time-start_time);
	free(copy_before);
	free(copy_after);
	free_t(T);
	omp_destroy_lock(&writelock);

}
