#include "MicroGlut.h"
#include "GL_utilities.h"
#include <math.h>
#include <loadobj.h>
#include "VectorUtils3.h"

GLfloat vertices[] = { -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f, -0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f, -0.5f, 0.5f,
    -0.5f, 0.5f, -0.5f, -0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f,
    0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, -0.5f, 0.5f,
    0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, -0.5f,
    -0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f, -0.5f, 0.5f, -0.5f, -0.5f,
    0.5f, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, -0.5f,
    -0.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f, -0.5f, -0.5f, 0.5f, 0.5f,
    0.5f, 0.5f, 0.5f, -0.5f, 0.5f, 0.5f, 0.5f, -0.5f, 0.5f };

GLfloat colors[] = { 0.583f, 0.771f, 0.014f, 0.609f, 0.115f, 0.436f, 0.327f,
    0.483f, 0.844f, 0.822f, 0.569f, 0.201f, 0.435f, 0.602f, 0.223f, 0.310f,
    0.747f, 0.185f, 0.597f, 0.770f, 0.761f, 0.559f, 0.436f, 0.730f, 0.359f,
    0.583f, 0.152f, 0.483f, 0.596f, 0.789f, 0.559f, 0.861f, 0.639f, 0.195f,
    0.548f, 0.859f, 0.014f, 0.184f, 0.576f, 0.771f, 0.328f, 0.970f, 0.406f,
    0.615f, 0.116f, 0.676f, 0.977f, 0.133f, 0.971f, 0.572f, 0.833f, 0.140f,
    0.616f, 0.489f, 0.997f, 0.513f, 0.064f, 0.945f, 0.719f, 0.592f, 0.543f,
    0.021f, 0.978f, 0.279f, 0.317f, 0.505f, 0.167f, 0.620f, 0.077f, 0.347f,
    0.857f, 0.137f, 0.055f, 0.953f, 0.042f, 0.714f, 0.505f, 0.345f, 0.783f,
    0.290f, 0.734f, 0.722f, 0.645f, 0.174f, 0.302f, 0.455f, 0.848f, 0.225f,
    0.587f, 0.040f, 0.517f, 0.713f, 0.338f, 0.053f, 0.959f, 0.120f, 0.393f,
    0.621f, 0.362f, 0.673f, 0.211f, 0.457f, 0.820f, 0.883f, 0.371f, 0.982f,
    0.099f, 0.879f };

Model* m;

unsigned int bunnyVertexArrayObjID;
unsigned int bunnyVertexBufferObjID;
unsigned int bunnyIndexBufferObjID;
unsigned int bunnyNormalBufferObjID;
unsigned int bunnyTexCoordBufferObjID;


#define near 1.0
#define far 30.0
#define right 0.5
#define left -0.5
#define top 0.5
#define bottom -0.5
GLfloat projectionMatrix[] = {    2.0f*near/(right-left), 0.0f, (right+left)/(right-left), 0.0f,
                                            0.0f, 2.0f*near/(top-bottom), (top+bottom)/(top-bottom), 0.0f,
                                            0.0f, 0.0f, -(far + near)/(far - near), -2*far*near/(far - near),
                                            0.0f, 0.0f, -1.0f, 0.0f };
mat4 rot, trans, total;



// vertex array object
unsigned int vertexArrayObjID;
GLfloat myMatrix[] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };

GLuint vbo_triangle, vbo_triangle_colors;
GLint attribute_coord2d, attribute_v_color;
GLuint myTex;

void OnTimer(int value) {
  glutPostRedisplay();
  glutTimerFunc(20, &OnTimer, value);
}
GLuint program;
void init(void) {
  m = LoadModel("../bunnyplus.obj");

  trans = T(0, 0, -4);
  rot = Ry(0.0);
  total = Mult(rot, trans);
  // vertex buffer object, used for uploading the geometry
  unsigned int vertexBufferObjID;
  unsigned int colorBufferObjID;
  // Reference to shader program

  dumpInfo();

  // GL inits
  glClearColor(0.2, 0.2, 0.5, 0);
  glDisable(GL_DEPTH_TEST);
  printError("GL inits");

  // Load and compile shader
  program = loadShaders("lab1-1.vert", "lab1-1.frag");
  printError("init shader");

  // Upload geometry to the GPU:

  glGenVertexArrays(1, &bunnyVertexArrayObjID);
  glGenBuffers(1, &bunnyVertexBufferObjID);
  glGenBuffers(1, &bunnyIndexBufferObjID);
  glGenBuffers(1, &bunnyNormalBufferObjID);
  glGenBuffers(1, &bunnyTexCoordBufferObjID);

  glBindVertexArray(bunnyVertexArrayObjID);
  printError("load1");

  // VBO for vertex data
  glBindBuffer(GL_ARRAY_BUFFER, bunnyVertexBufferObjID);
  glBufferData(GL_ARRAY_BUFFER, m->numVertices * 3 * sizeof(GLfloat),
      m->vertexArray, GL_STATIC_DRAW);
  glVertexAttribPointer(glGetAttribLocation(program, "in_Position"), 3,
      GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(glGetAttribLocation(program, "in_Position"));

  printError("load2");
  // VBO for normal data
  glBindBuffer(GL_ARRAY_BUFFER, bunnyNormalBufferObjID);
  glBufferData(GL_ARRAY_BUFFER, m->numVertices * 3 * sizeof(GLfloat),
      m->normalArray, GL_STATIC_DRAW);
  glVertexAttribPointer(glGetAttribLocation(program, "in_Normal"), 3, GL_FLOAT,
      GL_FALSE, 0, 0);
  glEnableVertexAttribArray(glGetAttribLocation(program, "in_Normal"));
  printError("load3");


  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, bunnyIndexBufferObjID);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, m->numIndices * sizeof(GLuint),
      m->indexArray, GL_STATIC_DRAW);
  printError("load4");

  glUniformMatrix4fv(glGetUniformLocation(program, "rotMatrix"), 1, GL_TRUE,
      myMatrix);
  printError("load5");

  if (m->texCoordArray != NULL) {
    printError("load6");
    glBindBuffer(GL_ARRAY_BUFFER, bunnyTexCoordBufferObjID);
    printError("load7");
    glBufferData(GL_ARRAY_BUFFER, m->numVertices * 2 * sizeof(GLfloat),
        m->texCoordArray, GL_STATIC_DRAW);
    printError("load8");
    glVertexAttribPointer(glGetAttribLocation(program, "inTexCoord"), 2,
        GL_FLOAT, GL_FALSE, 0, 0);
    printError("load9");
    glEnableVertexAttribArray(glGetAttribLocation(program, "inTexCoord"));
    printError("load10");
  }

  LoadTGATextureSimple("../maskros512.tga", &myTex);
  printError("load11");
  glBindTexture(GL_TEXTURE_2D, myTex);
  printError("load12");

  glUniform1i(glGetUniformLocation(program, "texUnit"), 0);
  printError("load13");


  glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, GL_TRUE,
      projectionMatrix);

  glUniformMatrix4fv(glGetUniformLocation(program, "mdlMatrix"), 1, GL_TRUE, total.m);
}

void display(void) {
  printError("pre display");

  // clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glBindVertexArray(bunnyVertexArrayObjID); // Select VAO
  glDrawElements(GL_TRIANGLES, m->numIndices, GL_UNSIGNED_INT, 0L);

  myMatrix[0] = cos((GLfloat) glutGet(GLUT_ELAPSED_TIME) / 1000.0);
  myMatrix[8] = -sin((GLfloat) glutGet(GLUT_ELAPSED_TIME) / 1000.0);
  myMatrix[2] = sin((GLfloat) glutGet(GLUT_ELAPSED_TIME) / 1000.0);
  myMatrix[10] = cos((GLfloat) glutGet(GLUT_ELAPSED_TIME) / 1000.0);

  glUniformMatrix4fv(glGetUniformLocation(program, "rotMatrix"), 1, GL_TRUE,
      myMatrix);
  printError("display");
  glutSwapBuffers();
}

int main(int argc, char *argv[]) {
  glutInit(&argc, argv);
  glutInitContextVersion(3, 2);
  glutCreateWindow("GL3 white triangle example");
  glutDisplayFunc(display);
  init();
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
  glEnable(GL_DEPTH_TEST);
  glutTimerFunc(20, &OnTimer, 0);
  glutMainLoop();
  return 0;
}
