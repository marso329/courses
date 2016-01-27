
// This is not really C++-code but pretty plain C code, but we compile it
// as C++ so we can integrate with CUDA seamlessly.

// If you plan on submitting your solution for the Parallel Sorting Contest,
// please keep the split into main file and kernel file, so we can easily
// insert other data.
#include <stdio.h>
__device__ 
static void exchange(int *i, int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}


__global__
void bitonic_kernel(int *data, int k, int j)
{
	
int i = threadIdx.x + blockIdx.x*blockDim.x;



	int ixj=i^j; // Calculate indexing!
        if ((ixj)>i)
        {  
		if ((i&k)==0 && data[i]>data[ixj]) exchange(&data[i],&data[ixj]);
		if ((i&k)!=0 && data[i]<data[ixj]) exchange(&data[i],&data[ixj]);
        } 



}



// No, this is not GPU code yet but just a copy of the CPU code, but this
// is where I want to see your GPU code!
void bitonic_gpu(int *data, int N)
{
printf("hello");
  int *dev_data;
  int size = N * sizeof(int);


  cudaMalloc((void**)&dev_data, size);
  cudaMemcpy(dev_data, data, size, cudaMemcpyHostToDevice);

  dim3 dimBlock (min(N ,1024), 1);
  dim3 dimGrid (N / 1024  + 1, 1);

int j,k;
  for (k=2;k<=N;k=2*k) // Outer loop, double size for each step
  {
    for (j=k>>1;j>0;j=j>>1) // Inner loop, half size for each step
    {
  	bitonic_kernel<<<dimGrid, dimBlock>>>(dev_data, k, j);
  	cudaThreadSynchronize();
    }
  }


  cudaMemcpy(data, dev_data, size, cudaMemcpyDeviceToHost);

  cudaFree(dev_data);

}
