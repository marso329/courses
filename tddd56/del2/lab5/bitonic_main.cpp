
// This is not really C++-code but pretty plain C code, but we compile it
// as C++ so we can integrate with CUDA seamlessly.

#ifdef GPU
#include "bitonic_kernel.h"
#endif
#include <stdio.h>
#include "milli.h"
#include <time.h>  
#include <stdlib.h> 

#ifdef K
#define SIZE K
#else
#define SIZE 16
#endif

#ifdef PRINT
#define MAXPRINTSIZE PRINT
#else
#define MAXPRINTSIZE 32
#endif
//int data[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
//int data2[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data[SIZE];
int data2[SIZE];

static void exchange(int *i, int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

void bitonic_cpu(int *data, int N)
{
  int i,j,k;
  for (k=2;k<=N;k=2*k) // Outer loop, double size for each step
  {
    for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
    {
      for (i=0;i<N;i++) // Loop over data
      {
        int ixj=i^j; // Calculate indexing!
        if ((ixj)>i)
        {
          if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
          if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
        }
      }
    }
  }
}

int main()
{
	srand (time(0));
	
	int temp;

for (int i =0;i<SIZE;i++){
	temp=rand() % 10000 + 1;
	data[i]=temp;
	data2[i]=temp;

}
  ResetMilli();
  bitonic_cpu(data, SIZE);
  printf("CPU:%f\n", GetSeconds());
  ResetMilli();
  #ifdef GPU
  bitonic_gpu(data2, SIZE);
  printf("GPU:%f\n", GetSeconds());

  for (int i=0;i<SIZE;i++)
    if (data[i] != data2[i])
    {
      printf("Error at %d ", i);
      return(1);
    }
 #endif
  // Print result
  if (SIZE <= MAXPRINTSIZE)
    for (int i=0;i<SIZE;i++)
      printf("%d ", data[i]);
  printf("\nYour sorting looks correct!\n");
}
