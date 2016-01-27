// Reduction lab, find maximum

#include <stdio.h>
#include "milli.c"
#define MAX_SHARED 12288

__global__ void find_max(int *data, int N)
{
  int i;
  i = threadIdx.x + blockDim.x*blockIdx.x;
  int elements=N;
  int first_pos,second_pos;
  int temp;
  if(i<elements/2){
  	int it=(elements)-1-i*2;
  	for (int j=0;j<it;j++){
  		if (j%2==0){
  		  first_pos=i*2;
  			second_pos=i*2+1;
  		}
  		else{
  		  first_pos=i*2+1;
  			second_pos=i*2+1+1;
  		}
  		if(data[first_pos]<data[second_pos]){
  			//swap
  			temp=data[first_pos];
  			data[first_pos]=data[second_pos];
  			data[second_pos]=temp;  		
  		}
			__syncthreads();

  	}
  	
  }

	// Write your CUDA kernel here
}


__global__ void find_max_opt(int *data, int N)
{
	 int i;
	 i = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int shared_data[2048];
	   int elements=N;
	 if(i<elements/2){
		shared_data[(i*2)%1024]=data[i*2];
		shared_data[(i*2+1)%1024]=data[i*2+1];
	}
	__syncthreads();
  int first_pos,second_pos;
  int temp;
  if(i<elements/2){
  	int it=(elements)-1-i*2;
  	for (int j=0;j<it;j++){
  		if (j%2==0){
  		  first_pos=i*2;
  			second_pos=i*2+1;
  		}
  		else{
  		  first_pos=i*2+1;
  			second_pos=i*2+1+1;
  		}
  		if(first_pos<blockIdx.x*1024 &&data[first_pos]<shared_data[second_pos%1024]){
  		  		temp=data[first_pos];
  					data[first_pos]=shared_data[second_pos%1024];
  					shared_data[second_pos%1024]=temp; 
  		}
  		else if(second_pos>=(blockIdx.x+1)*1024 && shared_data[first_pos%1024]<data[second_pos]){
  		  		temp=shared_data[first_pos%1024];
  					shared_data[first_pos%1024]=data[second_pos];
  					data[second_pos]=temp; 
  		}
  		else if(shared_data[first_pos%1024]<shared_data[second_pos%1024]){
  		  			temp=shared_data[first_pos%1024];
  			shared_data[first_pos%1024]=shared_data[second_pos%1024];
  			shared_data[second_pos%1024]=temp;  
  		}
  		
			__syncthreads();

		if(i==0 && j==it-1){
		data[0]=shared_data[0];
		printf("done\n");
		}
  	}
  	
  	
  }

	// Write your CUDA kernel here
}


void launch_cuda_kernel(int *data, int N)
{
	// Handle your CUDA kernel launches in this function
	int *devdata;
	int size = sizeof(int) * N;
	cudaMalloc( (void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice );
	
	// Dummy launch
	  dim3 dimBlock (min(N ,1024), 1);
  dim3 dimGrid (N / 1024  + 1, 1);
	find_max<<<dimGrid, dimBlock>>>(devdata, N);
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, N*sizeof(int), cudaMemcpyDeviceToHost ); 
	cudaFree(devdata);
/**	for (int i=0;i<N;i++){
	printf(" %i ",data[i]);
	}
	printf("\n");
**/

}

// CPU max finder (sequential)
void find_max_cpu(int *data, int N)
{
  int i, m;
  
	m = data[0];
	for (i=0;i<N;i++) // Loop over data
	{
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}

#define SIZE 1048576
//#define SIZE 16
// Dummy data in comments below for testing
int data[SIZE]; //= {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE]; //= {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
//int data[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3};
//int data2[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3};
int main()
{
  // Generate 2 copies of random data
  srand(time(NULL));
  for (long i=0;i<SIZE;i++)
  {
    data[i] = rand() % (SIZE * 5);
    data2[i] = data[i];
  }
  
  // The GPU will not easily beat the CPU here!
  // Reduction needs optimizing or it will be slow.
  ResetMilli();
  find_max_cpu(data, SIZE);
  printf("CPU time %f\n", GetSeconds());
  ResetMilli();
  launch_cuda_kernel(data2, SIZE);
  printf("GPU time %f\n", GetSeconds());

  // Print result
  printf("\n");
  printf("CPU found max %d\n", data[0]);
  printf("GPU found max %d\n", data2[0]);
}
