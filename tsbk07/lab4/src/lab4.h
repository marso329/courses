#ifndef LAB4_H
#define LAB4_H

//non-standard imports
#include "../lib/GL_utilities.h"
#include "../lib/loadobj.h"
#include "../lib/LoadTGA.h"
#include "../lib/MicroGlut.h"
#include "../lib/VectorUtils3.h"

//standard imports
#include <math.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <ncurses.h>
#include <time.h>
#include <iostream>
using namespace std;

//for rotation calculations
#define PI 3.14159265

//struct to store data about all models
struct ModelData {
	ModelData() :
			model(NULL), texture(1000), before(NULL), after(NULL),override(NULL),program(0),changed_direction(false),numbers_waited(0),texture_unit(GL_TEXTURE0) {

	}
	Model* model;
	vector<mat4*> transformations;
	GLuint texture;
	GLuint texture2;
	void (*before)(struct ModelData*);
	void (*after)(struct ModelData*);
	void (*override)(struct ModelData*);
	GLuint program;
	bool changed_direction;
	int numbers_waited;
	GLuint texture_unit;
	mat4* height_control;
	mat4* rotation_control;
	mat4* movement;
	vec3 direction;

};
typedef struct ModelData ModelData;

struct WorldData{
	mat4 projectionMatrix;
	mat4  camMatrix;
	vec3 cam ;
	vec3 lookAtPoint ;
	vec3 up;
	mat4 total;
	float mouse_x_old;
	float mouse_y_old;
	float sphere_radius;
	float distance_from_ground;
};
typedef struct WorldData WorldData;

//functions

Model* GenerateTerrain(TextureData *tex);
void OnTimer(int value);
void mouse_reset(int button, int state, int x, int y);
void mouse_motion_func(int x, int y) ;
mat4* create_matrix(mat4 matrix);
void enable_depth(ModelData*) ;
void disable_depth(ModelData*);
void enable_multitex(ModelData*);
void disable_multitex(ModelData*) ;
void mouse_motion_func(int x, int y,WorldData* worlddata);
GLfloat get_height(int x,int y,Model* model);
GLfloat get_height2(float x,float y,Model* model);
void GenerateNormals(Model* model);
float calculate_height(float x, float z, Model* model);
vec3 getNormal(unsigned int x, unsigned int z,Model* model);
#endif /* LAB4_H */
