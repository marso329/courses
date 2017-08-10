/*
 * graphics.h
 *
 *  Created on: Apr 9, 2016
 *      Author: martin
 */

#ifndef INCLUDES_GRAPHICS_H_
#define INCLUDES_GRAPHICS_H_

#define __GL_SYNC_TO_VBLANK 1
//non-standard imports
#include "GL_utilities.h"
#include "loadobj.h"
#include "LoadTGA.h"
#include "MicroGlut.h"
#include <opencv2/opencv.hpp>
#include "VectorUtils3.h"
#include "file_structures.h"
#include "support_functions.h"
#include <X11/XKBlib.h>
#include <boost/filesystem.hpp>
#include "constants.h"

//standard imports
#include <math.h>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <iostream>
#include <functional>
#include <X11/keysym.h>
#include <map>
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <regex>
#include <iterator>
class Renderer;
class Cube;
class Letter;

#define  GL_GLEXT_PROTOTYPES
typedef void (Renderer::*DisplayFunc)(Cube*);

class Cube {
public:
	~Cube();
	Model* model;
	std::vector<mat4*> transformations;
	GLuint texture;
	GLuint program;
	GLuint texture_unit;
	GLuint vertexbuffer;
	GLuint vao;
	DisplayFunc display_function;
	Renderer* master;
	std::vector<Letter*>* letters=NULL;
	float x_pos, y_pos;
	GLuint skybox_texture;
	Inode* represent = NULL;
	 GLfloat g_vertex_buffer_data[6];
	 bool line=false;
	 clock_t time_created=0.0;
	 bool shader_changed=false;
};

class Letter: public Cube{
public:
	~Letter();
};

class Renderer {
public:
	//constructor
	Renderer(Directory*);
	//destructor
	~Renderer();
	//initializes opengl
	void init();
	//display function
	void display();
	//render function for floor
	void display_floor(Cube* it);
	//render function for all inodes objects
	void display_inodes(Cube* it);
	//render function for all letters
	void display_letter(Cube* element);
	//render function for skydome
	void display_skybox(Cube* element);
	//creates the floor elements
	void create_floor();
	//creates all the inodes objects
	void create_inodes();
	//creates the skydome
	void create_skybox();
	//load any image opencv can handle and creates a standard 2d texture from it, texture generation is done in the function
	void load_image_cv(const char* filename, unsigned int* texture,
			int rotation = 0);
	//loads a tga file and creates a texture from it
	void load_image_tga(const char* filename, unsigned int* texture);
	//updates the cubes letter regarding the camera position
	void update_letters(Cube*);
	//mouse function, used to look around
	void mouse(int x, int y);
	//mouse function, used to detect clicks
	void mouse_press(int button, int state, int x, int y);
	//creates objects from all available letters
	void init_letters();
	//timer function to keep fps constant
	void timer(int);
	//keyboard function
	void keyboard(unsigned int key, int x, int y, int state);
	//creates a vector of letter objects from a string, the first vec3 is the center position and the second is the direction for seeing the letters
	std::vector<Letter*>* create_text(std::string, vec3, vec3);
	//returns the objects center position from the camera
	float distance_to_camera(Cube* data_cube);
	//when we are in promt mode this function is called from keyboard() and checks the inputs and adds it to promt_string
	void handle_promt(unsigned int key, int state);
	//called in display() to dislpay the current promt_string
	void draw_promt();
	//when enter is pressed in promt_mode this function is called to parse the promt_string
	void parse_promt();
	//called in display to add the cwd in the upper right corner
	void draw_cwd();
	void add_parse_text(char text);
	//called in display for each inode to check for collision
	void check_for_collision(Cube* element);
	void delete_model(Model* model);
	void init_new_directory();
	void subinit();
	void pick(int x,int y);
	void create_line(vec3 start,vec3 end);
	void display_line(Cube* element);
	void check_for_color_change(Cube* element);
	void list_files();
	void select_file(std::string filename);
	void help();
	void cd(std::string filename);
	void rename(std::string filename,std::string new_name);
	void delete_file(std::string filename);
	//the current directory
	Directory* cwd;
	//contains all the elements in the world
	std::vector<Cube*>* elements;
	//loads cubeplus in init and is used for all inodes objects
	Model* cube_model;
	//reference for deleting
	Model* skybox_model;
	//stores the position of the camera
	vec3 camera;
	//stores att which position we are looking
	vec3 looking;
	//always 0,1,0
	vec3 up;
	//used to keep track on if the mouse button is pressed or released
	bool button_released = true;
	//used to store the time last time the button was pressed(for doubleclick)
	clock_t time_button = 0;
	//used to store the mouse position from last iteration to know how much to move the looking_pos
	int mouse_x = 0;
	int mouse_y = 0;
	//store for if we are in promt mode
	bool promt_mode = false;
	//maps all printable characters to objects for them
	std::map<std::string, Model*>* letters;
	//used to store all characters we can print in the promt
	std::vector<char> printable;
	//stores the current promt_string
	std::string promt_string;
	std::string data_string;
	std::vector<std::string> promt_vector;
	//if we collide with a directory this is set to true so in the beginning of next iteration we change folder
	bool change_directory = false;
	//used to store the new folder if we change folder
	Directory* new_directory=NULL;
	mat4 frustum_matrix;
	Cube* selected=NULL;

};

#endif /* INCLUDES_GRAPHICS_H_ */
