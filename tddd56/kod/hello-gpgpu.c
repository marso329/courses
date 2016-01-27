// Hello World in a shader.
// Kind of twisted, since it uses signed chars.
// New, ARB/EXT-free version. Not strict GL3+ yet though.

#include <stdio.h>
#include <stdlib.h>
#include <OpenGL/gl.h>
#include <GLUT/glut.h>
#include <sys/times.h>

// Compile shaders. Old junk code, I have better!
GLuint setupShader(const GLchar **vertSrc, const GLchar **fragSrc)
{
	GLuint	programObject;	// the program used to update
	GLuint	fragmentShader, vertexShader;
	
	programObject = glCreateProgram();

	if (fragSrc != NULL)
	{	
		fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragmentShader, 1, fragSrc, NULL);
		glCompileShader(fragmentShader);
		glAttachShader(programObject, fragmentShader);
	}

	if (vertSrc != NULL)
	{	
		vertexShader = glCreateShader(GL_VERTEX_SHADER);
		glShaderSource(vertexShader, 1, vertSrc, NULL);
		glCompileShader(vertexShader);
		glAttachShader(programObject, vertexShader);
	}

	glLinkProgram(programObject);
	GLint progLinkSuccess;
	glGetProgramiv(programObject, GL_LINK_STATUS,
		&progLinkSuccess);
	if (!progLinkSuccess)
	{
		fprintf(stderr, "Shader could not be linked\n");
		exit(1);
	}
	return programObject;
}

// Add offset (texUnit2) to string (texUnit)
// Negative values end up as > 0.5, adjust them!
static const char *fragSource =
{
"uniform sampler2D texUnit;"
"uniform sampler2D texUnit2;"
"void main(void)"
"{"
"   vec4 texVal  = texture2D(texUnit, gl_TexCoord[0].xy);"
"   vec4 texVal2  = texture2D(texUnit2, gl_TexCoord[0].xy);"
"   if (texVal2.r > 0.5) texVal2.r -= 1.0;"
"   if (texVal2.g > 0.5) texVal2.g -= 1.0;"
"   if (texVal2.b > 0.5) texVal2.b -= 1.0;"
"   if (texVal2.a > 0.5) texVal2.a -= 1.0;"
"   gl_FragColor = texVal + texVal2;"
"}"
};

// Vertex shader, pass position and texcoord
char *vs =
{
"void main()"
"{"
"   gl_Position = ftransform();"
"   gl_TexCoord[0] = gl_MultiTexCoord0;"
"}"
};

int main(int argc, char **argv)
{
    // declare texture size, the actual data will be a vector 
    // of size texSize*texSize*4
	// Obs att texSize måste vara 2-potens på många (äldre) grafikkort!
    int texSize = 16;
#define N 16
    int i;
    // create test data
	char a[N] = "Hello \0\0\0\0\0\0\0\0\0\0";
	char b[N] = {15, 10, 6, 0, -12, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    // set up glut to get valid GL context and 
    // get extension entry points
    glutInit (&argc, argv);
    glutCreateWindow("TEST1");
//    glewInit();
    // viewport transform for 1:1 pixel=texel=data mapping
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0.0,texSize,0.0,texSize);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0,0,texSize,texSize);
    // create FBO and bind it (that is, use offscreen render target)
    GLuint fb;
    glGenFramebuffers(1,&fb); 
    glBindFramebuffer(GL_FRAMEBUFFER,fb);
    
    // create string texture
    GLuint tex;
	glActiveTexture(GL_TEXTURE0);
    glGenTextures (1, &tex);
    glBindTexture(GL_TEXTURE_2D,tex);
    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, 
                    GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, 
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // define texture NOT with floating point format (see process-array for FP example)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,
                 texSize/4,1,0,GL_RGBA,GL_UNSIGNED_BYTE, a); // Var tom textur i originalet

    // create offset texture
    GLuint offtex;
	glActiveTexture(GL_TEXTURE1);
    glGenTextures (1, &offtex);
    glBindTexture(GL_TEXTURE_2D,offtex);
    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, 
                    GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, 
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // define texture NOT with floating point format (see process-array for FP example)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,
                 texSize/4,1,0,GL_RGBA,GL_UNSIGNED_BYTE, b); // Var tom textur i originalet

    // create destination texture
    GLuint desttex;
    glGenTextures (1, &desttex);
    glBindTexture(GL_TEXTURE_2D,desttex);
    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, 
                    GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, 
                    GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // define texture NOT with floating point format (see process-array for FP example)
    glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,
                 texSize/4,1,0,GL_RGBA,GL_UNSIGNED_BYTE, NULL);

    // attach texture
    glFramebufferTexture2D(GL_FRAMEBUFFER, 
                              GL_COLOR_ATTACHMENT0, 
                              GL_TEXTURE_2D,desttex,0);

// Compile shader
	GLuint shader;
	shader = setupShader(&vs, &fragSource);
	glUseProgram(shader);
// Install input
	GLint loc = glGetUniformLocation(shader, "texUnit");
	glUniform1i(loc, 0); // texture unit 0
	loc = glGetUniformLocation(shader, "texUnit2");
	glUniform1i(loc, 1); // texture unit 1

// draw

// Make sure input textures are bound
	glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D,tex);
	glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D,offtex);
 
		glUseProgram(shader);
		glBegin(GL_QUADS);
		glTexCoord2f(0, 0);
		glVertex2f(0, 0);
		glTexCoord2f(1, 0);
		glVertex2f(texSize/4, 0);
		glTexCoord2f(1, 1);
		glVertex2f(texSize/4, texSize/4);
		glTexCoord2f(0, 1);
		glVertex2f(0, texSize/4);
		glEnd();
		glFlush();

	  printf("%s",a);

    // and read back
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glReadPixels(0, 0, texSize, 1, GL_RGBA,GL_UNSIGNED_BYTE,a);
	
    // print out results
    printf("%s\n",a);
    sleep(1);
    exit(0);
}
