//non-standard imports
#include "../lib/GL_utilities.h"
#include "../lib/loadobj.h"
#include "../lib/LoadTGA.h"
#include "../lib/MicroGlut.h"
#include "../lib/VectorUtils3.h"

//standard imports
#include <math.h>
#include <vector>
//#include <GLFW/glfw3.h>
#include <stdio.h>
#include <string.h>

#define PI 3.14159265

//obviously
using namespace std;

struct ModelData {
	ModelData() :
			model(NULL), texture(1000), before(NULL), after(NULL) {

	}
	Model* model;
	vector<mat4*> transformations;
	GLuint texture;
	void (*before)(struct ModelData*);
	void (*after)(struct ModelData*);
};
typedef struct ModelData ModelData;

vector<ModelData> all_models;
mat4 *rotation;
mat4* mill_trans;
mat4* skybox_matrix;

#define near 1.0
#define far 200.0
#define right 0.5
#define left -0.5
#define top 0.5
#define bottom -0.5
GLfloat projectionMatrix[] = { 2.0f * near / (right - left), 0.0f,
		(right + left) / (right - left), 0.0f, 0.0f, 2.0f * near
				/ (top - bottom), (top + bottom) / (top - bottom), 0.0f, 0.0f,
		0.0f, -(far + near) / (far - near), -2 * far * near / (far - near),
		0.0f, 0.0f, -1.0f, 0.0f };
bool button_been_pressed = false;
float mouse_x_old, mouse_y_old;
mat4 lookMatrix;
GLuint grass_texture;
GLuint maskros_texture;
GLuint skybox_texture;
GLuint dirt_texture;
GLuint program;
int roomsize = 30;
vec3 center = vec3(0, 10, 0);
vec3 camera_pos = vec3(roomsize, roomsize, roomsize);
vec3 camera_up = vec3(0, 1, 0);
vec3 trans_vector;


Point3D lightSourcesColorsArr[] = { {1.0f, 0.0f, 0.0f}, // Red light
                                 {0.0f, 1.0f, 0.0f}, // Green light
                                 {0.0f, 0.0f, 1.0f}, // Blue light
                                 {1.0f, 1.0f, 1.0f} }; // White light

Point3D lightSourcesDirectionsPositions[] = { {10.0f, 5.0f, 0.0f}, // Red light, positional
                                       {0.0f, 5.0f, 10.0f}, // Green light, positional
                                       {-1.0f, 0.0f, 0.0f}, // Blue light along X
                                       {0.0f, 0.0f, -1.0f} }; // White light along Z

GLfloat specularExponent[] = {10.0, 20.0, 60.0, 5.0};
GLint isDirectional[] = {0,0,1,1};

void OnTimer(int value) {
	glutPostRedisplay();
	glutTimerFunc(20, &OnTimer, value);
}

void mouse_reset(int button, int state, int x, int y) {
	if (state == 1) {
		button_been_pressed = false;
	}
}

void mouse_motion_func(int x, int y) {
	int window_height = get_height();
	int window_width = get_width();
	int mouse_x = x;
	int mouse_y = y;
	if (mouse_x < 0) {
		mouse_x = 0;
	}
	if (mouse_y < 0) {
		mouse_y = 0;
	}
	if (mouse_x > window_width) {
		mouse_x = window_width;
	}
	if (mouse_y > window_height) {
		mouse_y = window_width;
	}
	float rel_x = (float) mouse_x / (float) window_width;
	float rel_y = (float) mouse_y / (float) window_height;
	if (get_mouse_button() == 1) {
		float x = float((float) roomsize - center.x)
				* cos(rel_x * 360 * PI / 180.0) * sin(rel_y * 180 * PI / 180.0);
		float z = float((float) roomsize - center.y)
				* sin(rel_x * 360 * PI / 180.0) * sin(rel_y * 180 * PI / 180.0);
		float y = -float((float) roomsize - center.x)
				* cos(rel_y * 180 * PI / 180.0);
		camera_pos = vec3(x, y, z);
		lookMatrix = lookAtv(camera_pos, center, camera_up);

	}
	if (get_mouse_button() == 2) {
		if (!button_been_pressed) {
			mouse_x_old = rel_x;
			mouse_y_old = rel_y;
			button_been_pressed = true;
			vec3 look_dir = VectorSub(center, camera_pos);
			look_dir = Normalize(look_dir);
			trans_vector = CrossProduct(look_dir, Normalize(camera_up));
			return;
		}

		float mov = (mouse_x_old - rel_x) * 100.0;
		vec3 movement = vec3(mov * trans_vector.x, mov * trans_vector.y,
				mov * trans_vector.z);
		camera_pos = VectorAdd(camera_pos, movement);
		center = VectorAdd(center, movement);
		lookMatrix = lookAtv(camera_pos, center, camera_up);
		mouse_x_old = rel_x;
		mouse_y_old = rel_y;
		return;

	}

}

mat4* create_matrix(mat4 matrix) {
	mat4 *temp = (mat4*) malloc(sizeof(mat4));
	memcpy(temp->m, &matrix.m, sizeof(GLfloat) * 16);
	return temp;

}
void enable_depth(ModelData*) {
	glEnable(GL_DEPTH_TEST);
}
void disable_depth(ModelData*) {
	glDisable(GL_DEPTH_TEST);
}

void enable_multitex(ModelData*){
	glUniform1i(glGetUniformLocation(program, "multitexture"), 1);
}
void disable_multitex(ModelData*){
	glUniform1i(glGetUniformLocation(program, "multitexture"), 0);
}


Model* skybox;

void init(void) {
	dumpInfo();
	glewExperimental = GL_TRUE;
	glewInit();
	//clear the error which glewinit() throws,apperently it is normal
	while (glGetError() != GL_NO_ERROR) {
	}
	LoadTGATextureSimple((char*) "resources/grass.tga", &grass_texture);
	LoadTGATextureSimple((char*) "resources/maskros512.tga", &maskros_texture);
	LoadTGATextureSimple((char*) "resources/SkyBox512.tga", &skybox_texture);
	LoadTGATextureSimple((char*) "resources/dirt.tga", &dirt_texture);

	skybox = LoadModelPlus((char*) "resources/skybox.obj");

	rotation = (mat4*) malloc(sizeof(mat4));
	mill_trans = (mat4*) malloc(sizeof(mat4));
	skybox_matrix = (mat4*) malloc(sizeof(mat4));
	mat4 temp_matrix = T(0, 0, 0);
	memcpy(&mill_trans->m, &temp_matrix.m, sizeof(GLfloat) * 16);
	ModelData windmillmodel;
	windmillmodel.model = LoadModelPlus(
			(char*) "resources/windmill/windmill-walls.obj");
	windmillmodel.transformations.push_back(mill_trans);
	all_models.push_back(windmillmodel);

	ModelData windmillmodel1;
	windmillmodel1.model = LoadModelPlus(
			(char*) "resources/windmill/windmill-balcony.obj");
	windmillmodel1.transformations.push_back(mill_trans);
	all_models.push_back(windmillmodel1);

	ModelData windmillmodel2;
	windmillmodel2.model = LoadModelPlus(
			(char*) "resources/windmill/windmill-roof.obj");
	windmillmodel2.transformations.push_back(mill_trans);
	all_models.push_back(windmillmodel2);

	ModelData blade1;
	blade1.model = LoadModelPlus((char*) "resources/windmill/blade.obj");
	blade1.transformations.push_back(create_matrix(Ry(3.14)));
	blade1.transformations.push_back(create_matrix(T(4.6, 0, 0)));
	blade1.transformations.push_back(rotation);
	blade1.transformations.push_back(create_matrix(T(0, 9.3, 0)));
	blade1.transformations.push_back(mill_trans);
	all_models.push_back(blade1);

	ModelData blade2;
	blade2.model = LoadModelPlus((char*) "resources/windmill/blade.obj");
	blade2.transformations.push_back(create_matrix(Ry(3.14)));
	blade2.transformations.push_back(create_matrix(T(4.7, 0, 0)));
	blade2.transformations.push_back(create_matrix(Rx(3.14)));
	blade2.transformations.push_back(rotation);
	blade2.transformations.push_back(create_matrix(T(0, 9.2, 0)));
	blade2.transformations.push_back(mill_trans);
	all_models.push_back(blade2);

	ModelData blade3;
	blade3.model = LoadModelPlus((char*) "resources/windmill/blade.obj");
	blade3.transformations.push_back(create_matrix(Ry(3.14)));
	blade3.transformations.push_back(create_matrix(T(4.7, 0, 0)));
	blade3.transformations.push_back(create_matrix(Rx(1.57)));
	blade3.transformations.push_back(rotation);
	blade3.transformations.push_back(create_matrix(T(0, 9.25, 0)));
	blade3.transformations.push_back(mill_trans);
	all_models.push_back(blade3);

	ModelData blade4;
	blade4.model = LoadModelPlus((char*) "resources/windmill/blade.obj");
	blade4.transformations.push_back(create_matrix(Ry(3.14)));
	blade4.transformations.push_back(create_matrix(T(4.7, 0, 0)));
	blade4.transformations.push_back(create_matrix(Rx(-1.57)));
	blade4.transformations.push_back(rotation);
	blade4.transformations.push_back(create_matrix(T(0, 9.25, 0)));
	blade4.transformations.push_back(mill_trans);
	all_models.push_back(blade4);

	//ground
	ModelData ground;
	ground.model = LoadModelPlus((char*) "resources/ground.obj");
	ground.texture = grass_texture;
	ground.before=&enable_multitex;
	ground.after=&disable_multitex;
	all_models.push_back(ground);
	glActiveTexture(GL_TEXTURE1);
	glBindTexture(GL_TEXTURE_2D, dirt_texture);

	ModelData bunny;
	Model* temp_model = LoadModelPlus((char*) "resources/bunnyplus.obj");
	for (unsigned int j=0;j<20;j++){
	for (unsigned int i = 0; i < 20; i++) {
		bunny = ModelData();
		bunny.model = temp_model;
		bunny.texture = maskros_texture;
		bunny.transformations.push_back(create_matrix(Ry(1.57)));
		bunny.transformations.push_back(create_matrix(S(4, 4, 4)));
		bunny.transformations.push_back(create_matrix(T((j*10.0-100.0), 2.1, 15 + 5 * i)));
		all_models.push_back(bunny);
	}
	}
	for(unsigned int j=0;j<20;j++){
	for (unsigned int i = 0; i < 20; i++) {
		bunny = ModelData();
		bunny.model = temp_model;
		bunny.texture = maskros_texture;
		bunny.transformations.push_back(create_matrix(Ry(1.57)));
		bunny.transformations.push_back(create_matrix(S(4, 4, 4)));
		bunny.transformations.push_back(
				create_matrix(T((j*10.0-100.0), 2.1, -(float) (15 + 5 * i))));
		all_models.push_back(bunny);
	}}
	lookMatrix = lookAtv(camera_pos, center, camera_up);
	glClearColor(0.2, 0.2, 0.5, 0);
	//glDisable(GL_DEPTH_TEST);
	program = loadShaders("shaders/lab1-1.vert", "shaders/lab1-1.frag");
	glUniform1i(glGetUniformLocation(program, "texUnit"), 0);
	glUniform1i(glGetUniformLocation(program, "texUnit2"), 1);
	printError("init shader");

}
void update_keys() {
	if (glutKeyIsDown('w')) {
		printf("hello\n");
	}
}

void display(void) {
	update_keys();
	printError("pre display");
	//glBindTexture(GL_TEXTURE_2D, grass_texture);
	printError("GL inits1-4");
	// lookMatrix=lookAt(sin(glutGet(GLUT_ELAPSED_TIME) / 1000.0)*3.0,2,cos(glutGet(GLUT_ELAPSED_TIME) / 1000.0)*3,0,0,0,0,1,0);
	mat4 temp_trans;
	mat4 temp_matrix = Rx(glutGet(GLUT_ELAPSED_TIME) / 500.0);
	memcpy(&rotation->m, &temp_matrix.m, sizeof(GLfloat) * 16);

	//skybox stuff
	glUniform1i(glGetUniformLocation(program, "skybox"), 1);
	glUniform1i(glGetUniformLocation(program, "multitexture"), 0);
	//copy the lookmatrix to skybox matrix
	memcpy(&skybox_matrix->m, &lookMatrix.m, sizeof(GLfloat) * 16);
	//zero translation
	skybox_matrix->m[3] = 0;
	skybox_matrix->m[7] = 0;
	skybox_matrix->m[11] = 0;
	skybox_matrix->m[15] = 1;

	glDisable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUniformMatrix4fv(glGetUniformLocation(program, "lookMatrix"), 1, GL_TRUE,
			skybox_matrix->m);
	temp_matrix = T(0, -0.5, 0);
	glUniformMatrix4fv(glGetUniformLocation(program, "mdlMatrix"), 1, GL_TRUE,
			temp_matrix.m);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, skybox_texture);
	DrawModel(skybox, program, (char*) "in_Position", (char*) "in_Normal",
			(char*) "in_TexCoord");

	glEnable(GL_DEPTH_TEST);
	glUniform1i(glGetUniformLocation(program, "skybox"), 0);

	for (auto it = all_models.begin(); it != all_models.end(); it++) {
		if ((*it).before != NULL) {
			(*(*it).before)(&(*it));
		}
		if ((*it).texture != 1000) {
			glBindTexture(GL_TEXTURE_2D, (*it).texture);

		}
		temp_trans = T(0, 0, 0);
		glUniformMatrix4fv(glGetUniformLocation(program, "projMatrix"), 1,
		GL_TRUE, projectionMatrix);
		for (auto it2 = (*it).transformations.begin();
				it2 != (*it).transformations.end(); it2++) {
			temp_trans = Mult(*(*it2), temp_trans);
		}
		glUniformMatrix4fv(glGetUniformLocation(program, "mdlMatrix"), 1,
		GL_TRUE, temp_trans.m);
		glUniformMatrix4fv(glGetUniformLocation(program, "lookMatrix"), 1,
		GL_TRUE, lookMatrix.m);
		DrawModel((*it).model, program, (char*) "in_Position",
				(char*) "in_Normal", (char*) "in_TexCoord");
		if ((*it).after != NULL) {
			(*(*it).after)(&(*it));
		}
	}
	glUniform3fv(glGetUniformLocation(program, "lightSourcesDirPosArr"), 4, &lightSourcesDirectionsPositions[0].x);
	glUniform3fv(glGetUniformLocation(program, "lightSourcesColorArr"), 4, &lightSourcesColorsArr[0].x);
	glUniform1fv(glGetUniformLocation(program, "specularExponent"), 4, specularExponent);
	glUniform1iv(glGetUniformLocation(program, "isDirectional"), 4, isDirectional);

	glUniform3fv(glGetUniformLocation(program, "camera_position"), 1, &camera_pos.x);
	glutSwapBuffers();
}

int main(int argc, char *argv[]) {

	glutInit(&argc, argv);
	glutInitContextVersion(3, 2);
	glutCreateWindow((char*) "Its working......its working");

	glutDisplayFunc(display);
	init();
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glEnable(GL_DEPTH_TEST);
	glutTimerFunc(20, &OnTimer, 0);
	glutMouseFunc(&mouse_reset);
	glutMotionFunc(&mouse_motion_func);
	
	glutMainLoop();
	return 0;
}
