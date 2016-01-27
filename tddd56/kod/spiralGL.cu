// Small demo of CUDA-OpenGL interoperability.
// A number of vertices are animated, rotating around the center of the screen.
// By Ingemar Ragnemalm based on various online sources

#if defined (__APPLE__) || defined(MACOSX)
	#include <OpenGL/gl.h>
	#include <GLUT/glut.h>
#else
	#include <GL/glut.h>
#endif
#include <cuda_gl_interop.h>
#include <stdio.h>


#define USE_CPU 0

#define NUM_VERTS 64
#define kVarv 4

// Data. Could be GPU only if we didn't have an alternate data path for CPU processing
float dd[NUM_VERTS * 4];

GLuint positionsVBO;
struct cudaGraphicsResource* positionsVBO_CUDA;

// CUDA vertex kernel
__global__ void createVertices(float4* positions, float time, unsigned int num)
{ 
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	
	positions[x].w = 1.0;
	positions[x].z = 0.0;
	positions[x].x = 0.5*sin(kVarv * (time + x * 2 * 3.14 / num)) * x/num;
	positions[x].y = 0.5*cos(kVarv * (time + x * 2 * 3.14 / num)) * x/num;
}

// Same thing on CPU
void cpuVertices(float4* positions, float time, unsigned int num)
{
	unsigned int x;
	for (x = 0; x < num; x++)
	{
		positions[x].w = 1.0;
		positions[x].z = 0.0;
		positions[x].x = 0.5*sin(kVarv * (time + x * 2 * 3.14 / num)) * x/num;
		positions[x].y = 0.5*cos(kVarv * (time + x * 2 * 3.14 / num)) * x/num;
   }
   
   glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
   unsigned int size = NUM_VERTS * 4 * sizeof(float);
   glBufferData(GL_ARRAY_BUFFER, size, positions, GL_DYNAMIC_DRAW);
}

float anim = 0.0;

void display()
{
#ifdef USE_CPU
	cpuVertices((float4 *)&dd, anim, NUM_VERTS);
#else
	// Map buffer object for writing from CUDA
	float4* positions;

	cudaGraphicsMapResources(1, &positionsVBO_CUDA, 0);
	size_t num_bytes;
	cudaGraphicsResourceGetMappedPointer((void**)&positions, &num_bytes, positionsVBO_CUDA);
	// Execute kernel
	dim3 dimBlock(16, 1, 1);
	dim3 dimGrid(NUM_VERTS / dimBlock.x, 1, 1);
	createVertices<<<dimGrid, dimBlock>>>(positions, anim, NUM_VERTS);
	// Unmap buffer object
	cudaGraphicsUnmapResources(1, &positionsVBO_CUDA, 0);
#endif
	// Render from buffer object
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glColor4f(1,1,0,1);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);
	glDrawArrays(GL_POINTS, 0, NUM_VERTS);
	glDisableClientState(GL_VERTEX_ARRAY);
	anim += 0.01;
	
	// Swap buffers
	glutSwapBuffers();
}

void OnTimer(int value)
{
	glutPostRedisplay();
	glutTimerFunc(10, &OnTimer, value);
}

void deleteVBO()
{
	cudaGraphicsUnregisterResource(positionsVBO_CUDA);
	glDeleteBuffers(1, &positionsVBO);
}

int main(int argc, char **argv)
{
	 // Explicitly set device, ask for a device with compute capability 1.1 or better
	cudaDeviceProp  prop;
	int dev;
	memset( &prop, 0, sizeof( cudaDeviceProp ) );
	prop.major = 1;
	prop.minor = 1;
	cudaChooseDevice(&dev, &prop);
	cudaGLSetGLDevice(dev);
	printf("The device is %d\n", dev);
	
	// Initialize OpenGL and GLUT
 	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(400, 400);
	glutCreateWindow("Cuda simple GL Interop (VBO)");
	glutTimerFunc(20, &OnTimer, 0);
	
	glutDisplayFunc(display);
	// Create buffer object and register it with CUDA
	glGenBuffers(1, &positionsVBO);
	glBindBuffer(GL_ARRAY_BUFFER, positionsVBO);
	unsigned int size = NUM_VERTS * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, NULL, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGraphicsGLRegisterBuffer(&positionsVBO_CUDA, positionsVBO, cudaGraphicsMapFlagsWriteDiscard);
	
	// Launch rendering loop
	glutMainLoop();
}
