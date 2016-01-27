// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

static struct timeval timeStart;
static char hasStart = 0;

int GetMilliseconds();
int GetMicroseconds();
double GetSeconds();

// Optional setting of the start time. If these are not used,
// the first call to the above functions will be the start time.
void ResetMilli();
void SetMilli(int seconds, int microseconds);


int GetMilliseconds()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) / 1000 + (tv.tv_sec - timeStart.tv_sec)*1000;
}

int GetMicroseconds()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (tv.tv_usec - timeStart.tv_usec) + (tv.tv_sec - timeStart.tv_sec)*1000000;
}

double GetSeconds()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	if (!hasStart)
	{
		hasStart = 1;
		timeStart = tv;
	}
	return (double)(tv.tv_usec - timeStart.tv_usec) / 1000000.0 + (double)(tv.tv_sec - timeStart.tv_sec);
}

// If you want to start from right now.
void ResetMilli()
{
	struct timeval tv;

	gettimeofday(&tv, NULL);
	hasStart = 1;
	timeStart = tv;
}

// If you want to start from a specific time.
void SetMilli(int seconds, int microseconds)
{
	hasStart = 1;
	timeStart.tv_sec = seconds;
	timeStart.tv_usec = microseconds;
}
void add_matrix(float *a, float *b, float *c, int N)
{
	int index;
	
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main()
{
	const int N = 16;

	float a[N*N];
	float b[N*N];
	float c[N*N];

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	int start_time=GetMicroseconds();
	add_matrix(a, b, c, N);
	int end_time=GetMicroseconds();
	
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}
	printf("\n time elapsed is: %i \n",end_time-start_time);
}
