
#include "MicroGlut.h"
#include "GL_utilities.h"
#include <math.h>

 GLfloat vertices[] = {
     -0.5f,-0.5f,-0.5f, 
     -0.5f,-0.5f, 0.5f,
     -0.5f, 0.5f, 0.5f, 
     0.5f, 0.5f,-0.5f, 
     -0.5f,-0.5f,-0.5f,
     -0.5f, 0.5f,-0.5f, 
     0.5f,-0.5f, 0.5f,
     -0.5f,-0.5f,-0.5f,
     0.5f,-0.5f,-0.5f,
     0.5f, 0.5f,-0.5f,
     0.5f,-0.5f,-0.5f,
     -0.5f,-0.5f,-0.5f,
     -0.5f,-0.5f,-0.5f,
     -0.5f, 0.5f, 0.5f,
     -0.5f, 0.5f,-0.5f,
     0.5f,-0.5f, 0.5f,
     -0.5f,-0.5f, 0.5f,
     -0.5f,-0.5f,-0.5f,
     -0.5f, 0.5f, 0.5f,
     -0.5f,-0.5f, 0.5f,
     0.5f,-0.5f, 0.5f,
     0.5f, 0.5f, 0.5f,
     0.5f,-0.5f,-0.5f,
     0.5f, 0.5f,-0.5f,
     0.5f,-0.5f,-0.5f,
     0.5f, 0.5f, 0.5f,
     0.5f,-0.5f, 0.5f,
     0.5f, 0.5f, 0.5f,
     0.5f, 0.5f,-0.5f,
     -0.5f, 0.5f,-0.5f,
     0.5f, 0.5f, 0.5f,
     -0.5f, 0.5f,-0.5f,
     -0.5f, 0.5f, 0.5f,
     0.5f, 0.5f, 0.5f,
     -0.5f, 0.5f, 0.5f,
     0.5f,-0.5f, 0.5f
 };
 
GLfloat colors[] = {
     0.583f,  0.771f,  0.014f,
     0.609f,  0.115f,  0.436f,
     0.327f,  0.483f,  0.844f,
     0.822f,  0.569f,  0.201f,
     0.435f,  0.602f,  0.223f,
     0.310f,  0.747f,  0.185f,
     0.597f,  0.770f,  0.761f,
     0.559f,  0.436f,  0.730f,
     0.359f,  0.583f,  0.152f,
     0.483f,  0.596f,  0.789f,
     0.559f,  0.861f,  0.639f,
     0.195f,  0.548f,  0.859f,
     0.014f,  0.184f,  0.576f,
     0.771f,  0.328f,  0.970f,
     0.406f,  0.615f,  0.116f,
     0.676f,  0.977f,  0.133f,
     0.971f,  0.572f,  0.833f,
     0.140f,  0.616f,  0.489f,
     0.997f,  0.513f,  0.064f,
     0.945f,  0.719f,  0.592f,
     0.543f,  0.021f,  0.978f,
     0.279f,  0.317f,  0.505f,
     0.167f,  0.620f,  0.077f,
     0.347f,  0.857f,  0.137f,
     0.055f,  0.953f,  0.042f,
     0.714f,  0.505f,  0.345f,
     0.783f,  0.290f,  0.734f,
     0.722f,  0.645f,  0.174f,
     0.302f,  0.455f,  0.848f,
     0.225f,  0.587f,  0.040f,
     0.517f,  0.713f,  0.338f,
     0.053f,  0.959f,  0.120f,
     0.393f,  0.621f,  0.362f,
     0.673f,  0.211f,  0.457f,
     0.820f,  0.883f,  0.371f,
     0.982f,  0.099f,  0.879f
 };

// vertex array object
unsigned int vertexArrayObjID;
GLfloat myMatrix[] = {    1.0f, 0.0f, 0.0f, 0.0f,
                        0.0f, 1.0f, 0.0f, 0.0f,
                        0.0f, 0.0f, 1.0f, 0.0f,
                        0.0f, 0.0f, 0.0f, 1.0f };


GLuint vbo_triangle, vbo_triangle_colors;
GLint attribute_coord2d, attribute_v_color;




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
	printError("init arrays1");
	glBufferData(GL_ARRAY_BUFFER, 36*3*sizeof(GLfloat), vertices, GL_STATIC_DRAW);
	printError("init arrays2");
	glVertexAttribPointer(glGetAttribLocation(program, "in_Position"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	printError("init arrays3"); 
	glEnableVertexAttribArray(glGetAttribLocation(program, "in_Position"));
	glUniformMatrix4fv(glGetUniformLocation(program, "myMatrix"), 1, GL_TRUE, myMatrix);
	
	printError("init arrays4");
	
	
	glGenBuffers(1, &colorBufferObjID);
	glBindBuffer(GL_ARRAY_BUFFER, colorBufferObjID); 
	glBufferData(GL_ARRAY_BUFFER, 36*3*sizeof(GLfloat), colors, GL_STATIC_DRAW); 
	glVertexAttribPointer(glGetAttribLocation(program, "in_Color"), 3, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(glGetAttribLocation(program, "in_Color"));
printError("init arrays5");
}

void display(void)
{
	printError("pre display");

	// clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glBindVertexArray(vertexArrayObjID);	// Select VAO
	glDrawArrays(GL_TRIANGLES, 0, 36);	// draw object
	
		myMatrix[5] = cos((GLfloat)glutGet(GLUT_ELAPSED_TIME)/1000.0);
	myMatrix[6] = -sin((GLfloat)glutGet(GLUT_ELAPSED_TIME)/1000.0);
	myMatrix[9] = sin((GLfloat)glutGet(GLUT_ELAPSED_TIME)/1000.0);
	myMatrix[10] = cos((GLfloat)glutGet(GLUT_ELAPSED_TIME)/1000.0);
	glUniformMatrix4fv(glGetUniformLocation(program, "myMatrix"), 1, GL_TRUE, myMatrix);
	
	
	printError("display");
	glutSwapBuffers();
}

int main(int argc, char *argv[])
{
	glutInit(&argc, argv);
	glutInitContextVersion(3, 2);
	glutCreateWindow ("GL3 white triangle example");
	glutDisplayFunc(display); 
	init ();
		glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glEnable(GL_DEPTH_TEST);
	glutTimerFunc(20, &OnTimer, 0);
	glutMainLoop();
	return 0;
}
