#include "MicroGlut.h"
#include "GL_utilities.h"
#include <math.h>
#include <loadobj.h>
#include "VectorUtils3.h"


Model* m;
Model* teddy;

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

mat4 rot, trans, total,lookMatrix;
mat4 teddyrot,teddytrans,teddytotal;
GLuint myTex;


void OnTimer(int value) {
  glutPostRedisplay();
  glutTimerFunc(20, &OnTimer, value);
}

GLuint program;
void init(void) {
  dumpInfo();

  m = LoadModelPlus("../bunnyplus.obj");
  printError("GL inits1-1");
  teddy = LoadModelPlus("../cubeplus.obj");
  printError("GL inits1-2");
  LoadTGATextureSimple("../maskros512.tga", &myTex);
  printError("GL inits1-3");
  printError("GL inits1-5");

  trans = T(0, 0, -4);
  rot = Ry(0.0);
  total = Mult(rot, trans);
  teddytrans=T(0,1,-4);
  teddyrot=Ry(0.0);
  teddytotal=Mult(teddyrot,teddytrans);
  lookMatrix=lookAt(2,2,2,0,0,0,0,1,0);
  printError("GL inits2");
  // GL inits
  glClearColor(0.2, 0.2, 0.5, 0);
  glDisable(GL_DEPTH_TEST);
  printError("GL inits3");

  // Load and compile shader
  program = loadShaders("lab1-1.vert", "lab1-1.frag");
  printError("init shader");

}

void display(void) {
  printError("pre display");
  glBindTexture(GL_TEXTURE_2D, myTex);
  printError("GL inits1-4");
  lookMatrix=lookAt(sin(glutGet(GLUT_ELAPSED_TIME) / 1000.0)*3.0,2,cos(glutGet(GLUT_ELAPSED_TIME) / 1000.0)*3,0,0,0,0,1,0);

  trans = T(0, 0, 1);
  rot = Ry(glutGet(GLUT_ELAPSED_TIME) / 500.0);
  total = Mult( trans,rot);
  // clear the screen
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, GL_TRUE,
      projectionMatrix);

  glUniformMatrix4fv(glGetUniformLocation(program, "mdlMatrix"), 1, GL_TRUE, total.m);

  glUniformMatrix4fv(glGetUniformLocation(program, "lookMatrix"), 1, GL_TRUE, lookMatrix.m);

  DrawModel(m, program, "in_Position", "in_Normal", "in_TexCoord");
  trans = T(0, 0, -1);
  rot = Ry(-glutGet(GLUT_ELAPSED_TIME) / 1000.0);
  total = Mult( trans,rot);
  glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1, GL_TRUE,
      projectionMatrix);

  glUniformMatrix4fv(glGetUniformLocation(program, "mdlMatrix"), 1, GL_TRUE, total.m);

  glUniformMatrix4fv(glGetUniformLocation(program, "lookMatrix"), 1, GL_TRUE, lookMatrix.m);
  DrawModel(teddy, program, "in_Position", "in_Normal", "in_TexCoord");
  printError("display");
  glUniform1i(glGetUniformLocation(program, "texUnit"), 0);
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
