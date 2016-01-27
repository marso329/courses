// alotsimplerGL?
// Ingemars version av simpleGL, cut down to the bare essentials.
// Based on NVidia's example code


#include <stdio.h>
#include <string.h>
#include <math.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <cuda_gl_interop.h>

const unsigned int window_width = 512;
const unsigned int window_height = 512;
const unsigned int mesh_width = 256;
const unsigned int mesh_height = 256;


__global__ void kernel(float4* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	// calculate uv coordinates
	float u = x / (float) width;
	float v = y / (float) height;
	u = u*2.0f - 1.0f;
	v = v*2.0f - 1.0f;

	// calculate simple sine wave pattern
	float freq = 4.0f;
	float w = sinf(u*freq + time) * cosf(v*freq + time) * 0.5f;

	// write output vertex
	pos[y*width+x] = make_float4(u, w, v, 1.0f);
}

float anim = 0.0;
GLuint vbo;

void runCuda(struct cudaGraphicsResource **vbo_resource)
{
	// map OpenGL buffer object for writing from CUDA
	float4 *dptr;
	// DEPRECATED: cutilSafeCall(cudaGLMapBufferObject((void**)&dptr, vbo));
	cudaGraphicsMapResources(1, vbo_resource, 0);
	size_t num_bytes; 
	int err = cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes,  
							   *vbo_resource);
	if (err) printf("%d\n", err);

	// execute the kernel
	dim3 block(8, 8, 1);
	dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
	kernel<<< grid, block>>>(dptr, mesh_width, mesh_height, anim);

	// unmap buffer object
	cudaGraphicsUnmapResources(1, vbo_resource, 0);
}


void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
		   unsigned int vbo_res_flags)
{
	// create buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	
	// initialize buffer object
	unsigned int size = mesh_width * mesh_height * 4 * sizeof(float);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	
	// register this buffer object with CUDA
	cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags);
}

float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;
struct cudaGraphicsResource *cuda_vbo_resource;

void cpukernel()
{
	float4 *dptr;
	size_t size = mesh_width * mesh_height * 4 * sizeof(float);
//	cudaGraphicsResourceGetMappedPointer((void **)&dptr, &size,
//							   cuda_vbo_resource);

	unsigned int x;
	unsigned int y;

	for (x = 0; x < mesh_width; x++)
	for (y = 0; y < mesh_height; y++)
	{
		// calculate uv coordinates
		float u = x / (float) mesh_width;
		float v = y / (float) mesh_height;
		u = u*2.0f - 1.0f;
		v = v*2.0f - 1.0f;

		// calculate simple sine wave pattern
		float freq = 4.0f;
		float w = sinf(u*freq + anim) * cosf(v*freq + anim) * 0.5f;

		// write output vertex
		dptr[y*mesh_width+x] = make_float4(u, w, v, 1.0f);
	}
	// Upload to VBO!
//	unsigned int size = width * height * 4 * sizeof(float);
//	glBindBuffer(GL_ARRAY_BUFFER, vbo);
//	glBufferData(GL_ARRAY_BUFFER, size, pos, GL_DYNAMIC_DRAW);
}


void display()
{
	// run CUDA kernel to generate vertex positions
	runCuda(&cuda_vbo_resource);
//	cpukernel();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);

	glEnableClientState(GL_VERTEX_ARRAY);
	glColor3f(1.0, 0.0, 0.0);
	glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
	glutPostRedisplay();

	anim += 0.01;
}

int main(int argc, char **argv)
{
	cudaGLSetGLDevice(0);

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Cuda a lot simpler GL Interop (VBO)");

	// default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, window_width, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

	
	// register callbacks
	glutDisplayFunc(display);
//	glutKeyboardFunc(keyboard);
//	glutMouseFunc(mouse);
//	glutMotionFunc(motion);
	
	// create VBO
	createVBO(&vbo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

	cudaGLSetGLDevice( 0 );
	
	glutMainLoop();
	
	return true;
}
