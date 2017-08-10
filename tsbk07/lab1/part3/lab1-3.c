// Lab 1-1.
// This is the same as the first simple example in the course book,
// but with a few error checks.
// Remember to copy your file to a new on appropriate places during the lab so you keep old results.
// Note that the files "lab1-1.frag", "lab1-1.vert" are required.

// Should work as is on Linux and Mac. MS Windows needs GLEW or glee.
// See separate Visual Studio version of my demos.
#ifdef __APPLE__
	#include <OpenGL/gl3.h>
	// Linking hint for Lightweight IDE
	// uses framework Cocoa
#endif
#include "MicroGlut.h"
#include "GL_utilities.h"
#include <math.h>

#define PI 3.141592651

// Globals
// Data would normally be read from files
GLfloat vertices[] =
{
	-1.0f,-0.5f,0.0f,
	-0.0f,0.0f,0.0f,
	0.5f,-0.5f,0.0f
};

// vertex array object
unsigned int vertexArrayObjID;
GLfloat myMatrix[] = {    1.0f, 0.0f, 0.0f, 0.5f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f };


GLuint vbo_triangle, vbo_triangle_colors;
GLint attribute_coord2d, attribute_v_color;

GLfloat colors[] =
{
	1.0f,1.0f,1.0f,1.0f,
	0.5f,0.5f,0.5f,0.5f,
	0.0f,0.0f,0.0f,0.0f
};




void OnTimer(int value)
{
    glutPostRedisplay();
    glutTimerFunc(20, &OnTimer, value);
}
GLuint program;
void init(void)
{
	// vertex buffer object, used for uploading the geometry
	unsigned int vertexBufferObjID;
	unsigned int colorBufferObjID;
	// Reference to shader program

	dumpInfo();

	// GL inits
	glClearColor(0.2,0.2,0.5,0);
	glDisable(GL_DEPTH_TEST);
	printError("GL inits");

	// Load and compile shader
	program = loadShaders("lab1-1.vert", "lab1-1.frag");
	printError("init shader");
	
	// Upload geometry to the GPU:
	
	// Allocate and activate Vertex Array Object
	glGenVertexArrays(1, &vertexArrayObjID);
	glBindVertexArray(vertexArrayObjID);
	// Allocate Vertex Buffer Objects
	glGenBuffers(1, &vertexBufferObjID);
	
	// VBO for vertex data
	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID);
	glBufferData(GL_ARRAY_BUFFER, 9*sizeof(GLfloat), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer(glGetAttribLocation(program, "in_Position"), 3, GL_FLOAT, GL_FALSE, 0, 0); 
	glEnableVertexAttribArray(glGetAttribLocation(program, "in_Position"));

	glGenBuffers(1, &colorBufferObjID);
	glBindBuffer(GL_ARRAY_BUFFER, colorBufferObjID); 
	glBufferData(GL_ARRAY_BUFFER, 12*sizeof(GLfloat), colors, GL_STATIC_DRAW); 
	glVertexAttribPointer(glGetAttribLocation(program, "in_Color"), 4, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(glGetAttribLocation(program, "in_Color"));
	// End of upload of geometry
	glUniformMatrix4fv(glGetUniformLocation(program, "myMatrix"), 1, GL_TRUE, myMatrix);
	
	printError("init arrays");
}
GLfloat old_time;

void display(void)
{
	printError("pre display");

	// clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBindVertexArray(vertexArrayObjID);	// Select VAO
	glDrawArrays(GL_TRIANGLES, 0, 3);	// draw object
	printError("display");
	myMatrix[5] = cos((GLfloat)glutGet(GLUT_ELAPSED_TIME)/100.0);
	myMatrix[6] = -sin((GLfloat)glutGet(GLUT_ELAPSED_TIME)/100.0);
	myMatrix[9] = sin((GLfloat)glutGet(GLUT_ELAPSED_TIME)/100.0);
	myMatrix[10] = cos((GLfloat)glutGet(GLUT_ELAPSED_TIME)/100.0);
	glUniformMatrix4fv(glGetUniformLocation(program, "myMatrix"), 1, GL_TRUE, myMatrix);
	glutSwapBuffers();
	printf("time: %f \n",(GLfloat)glutGet(GLUT_ELAPSED_TIME)-old_time);
	old_time=(GLfloat)glutGet(GLUT_ELAPSED_TIME);
}

int main(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitContextVersion(3, 2);
	glutCreateWindow ("GL3 white triangle example");
	glutDisplayFunc(display); 
	init ();
	glutTimerFunc(20, &OnTimer, 0);
	glutMainLoop();
	return 0;
}
