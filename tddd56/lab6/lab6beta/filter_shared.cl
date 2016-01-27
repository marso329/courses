/*
 * Image filter in OpenCL
 */

#define KERNELSIZE 2

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m)
{ 
  unsigned int i = get_global_id(1) % 512;
  unsigned int j = get_global_id(0) % 512;
	
  
  int k, l;
  unsigned int sumx, sumy, sumz;
  
	
	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);
	
	if (j < n && i < m) // If inside image
	{
		if (i >= KERNELSIZE && i < m-KERNELSIZE && j >= KERNELSIZE && j < n-KERNELSIZE)
		{


	  unsigned int local_x = i%32+2;
	unsigned int local_y = j%32+2;
		__local unsigned char local_memory[36*36*3];

		//-2,-2
		local_memory[(local_x-2)*3+(local_y-2)*3*32+0]=image[((i-2)*n+(j-2))*3+0];
		local_memory[(local_x-2)*3+(local_y-2)*3*32+1]=image[((i-2)*n+(j-2))*3+1];
		local_memory[(local_x-2)*3+(local_y-2)*3*32+2]=image[((i-2)*n+(j-2))*3+2];
		
		//2,-2
		local_memory[(local_x+2)*3+(local_y-2)*3*32+0]=image[((i+2)*n+(j-2))*3+0];
		local_memory[(local_x+2)*3+(local_y-2)*3*32+1]=image[((i+2)*n+(j-2))*3+1];
		local_memory[(local_x+2)*3+(local_y-2)*3*32+2]=image[((i+2)*n+(j-2))*3+2];
		
		//-2,2
		local_memory[(local_x-2)*3+(local_y+2)*3*32+0]=image[((i-2)*n+(j+2))*3+0];
		local_memory[(local_x-2)*3+(local_y+2)*3*32+1]=image[((i-2)*n+(j+2))*3+1];
		local_memory[(local_x-2)*3+(local_y+2)*3*32+2]=image[((i-2)*n+(j+2))*3+2];
		
		//2,2
		local_memory[(local_x+2)*3+(local_y+2)*3*32+0]=image[((i+2)*n+(j+2))*3+0];
		local_memory[(local_x+2)*3+(local_y+2)*3*32+1]=image[((i+2)*n+(j+2))*3+1];
		local_memory[(local_x+2)*3+(local_y+2)*3*32+2]=image[((i+2)*n+(j+2))*3+2];
		
		barrier(CLK_LOCAL_MEM_FENCE);
		// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
				for(l=-KERNELSIZE;l<=KERNELSIZE;l++)	
				{

					
					sumx += local_memory[(local_x+k)*3+(local_y+l)*3*32+0];
					sumy += local_memory[(local_x+k)*3+(local_y+l)*3*32+1];
					sumz += local_memory[(local_x+k)*3+(local_y+l)*3*32+2];

				}
			out[(i*n+j)*3+0] = sumx/divby;
			out[(i*n+j)*3+1] = sumy/divby;
			out[(i*n+j)*3+2] = sumz/divby;
			
		}
		else
		// Edge pixels are not filtered
		{
			out[(i*n+j)*3+0] = image[(i*n+j)*3+0];
			out[(i*n+j)*3+1] = image[(i*n+j)*3+1];
			out[(i*n+j)*3+2] = image[(i*n+j)*3+2];
		}
	}
}
