

#include <stdio.h>

const int row=16;
const int N = row*row; 
const int blocksize = 16; 

__global__ 
void add_matrix(float *a,float *b,float *c) 
{
//calculate index 
	int x=blockDim.x*blockIdx.x+threadIdx.x;
	int y=blockDim.y*blockIdx.y+threadIdx.y;
	int index = y * gridDim.x * blockDim.x + x;
	//add
	c[index] = a[index] + b[index];
}

int main()
{
	//create matrices
	float *a = new float[N];
	float *b = new float[N];
	float *c = new float[N];
	
	//fill a and be with data
	for (int i =0;i<N;i++){
		a[i]=i;
		b[i]=i;
	}
	
	//pointers to data on cuda
	float *cuda_a;
	float *cuda_b;
	float *cuda_c;
	
	//allocat space on cuda
	cudaMalloc( (void**)&cuda_a, N*sizeof(float) );
	cudaMalloc( (void**)&cuda_b, N*sizeof(float) );
	cudaMalloc( (void**)&cuda_c, N*sizeof(float) );
	
	//copy data to cuda
	cudaMemcpy( cuda_a, a, N*sizeof(float), cudaMemcpyHostToDevice ); 
	cudaMemcpy( cuda_b, b, N*sizeof(float), cudaMemcpyHostToDevice ); 
	
	//set size
	dim3 dimBlock( blocksize, blocksize );
	dim3 dimGrid( 1, 1	 );
	
	
	//setup time measurement
	cudaEvent_t myEvent;
	cudaEvent_t laterEvent;
	cudaEventCreate(&laterEvent);
	cudaEventCreate(&myEvent);
	cudaEventRecord(myEvent, 0);
	cudaEventSynchronize(myEvent);	
	//start calculation
	add_matrix<<<dimGrid, dimBlock>>>(cuda_a,cuda_b,cuda_c);
	
	//sync
	cudaThreadSynchronize();
	cudaEventRecord(laterEvent, 0);
	cudaEventSynchronize(laterEvent);
	float theTime;
	cudaEventElapsedTime(&theTime, myEvent, laterEvent);
	
	//download results
	cudaMemcpy( c, cuda_c, N*sizeof(float), cudaMemcpyDeviceToHost );
	
	//free memory
	cudaFree( cuda_a );
	cudaFree( cuda_b );
	cudaFree( cuda_c );
	
	printf("\n");
	for (int i=0;i<N;i++){
	if (i%row==0){
	printf("\n");
	}
	printf(" %.2f ",c[i]);
	}
	printf("\n elasped time: %f \n",theTime);
	
	return EXIT_SUCCESS;
}
