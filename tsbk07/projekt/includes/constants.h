/*
 * constants.h
 *
 *  Created on: Apr 9, 2016
 *      Author: martin
 */

#ifndef INCLUDES_CONSTANTS_H_
#define INCLUDES_CONSTANTS_H_
#include "GL_utilities.h"
#include "loadobj.h"
#include "LoadTGA.h"
#include "MicroGlut.h"
#include "VectorUtils3.h"
const int FPS=30;

const float TEXTSIZE=0.1;
const float TEXTDISTANCE=0.2;

const float TIME_FOR_LINES_TO_DISAPPEAR =100000.0; //5 sec
//higher is decreased time between clicks
const int TIME_FOR_DOUBLECLICK=60000;
const float DISTANCE_TO_MOVE_KEYBOARD=0.09;
const float DISTANCE_TO_MOVE_MOUSE=0.05;
const float INODE_SPHERE_RADIUS=0.7;


static Point3D lightSourcesColorsArr[] = { { 1.0f, 1.0f, 1.0f }, // white light
		{ 1.0f, 1.0f, 1.0f }, // white light
		{ 1.0f, 1.0f, 1.0f }, // Blue light
		{ 1.0f, 1.0f, 1.0f } }; // White light

static Point3D lightSourcesDirectionsPositions[] = { { 10.0f, 10.0f, 0.0f }, // Red light, positional
		{ 0.0f, 10.0f, 10.0f }, // Green light, positional
		{ 0.0f, 10.0f, -10.0f }, // Blue light along X
		{ -10.0f, 10.0f, -1.0f } }; // White light along Z
__attribute__((unused))
static GLfloat specularExponent[] = { 10.0, 20.0, 60.0, 5.0 };
__attribute__((unused))
static GLint isDirectional[] = { 0, 0, 0, 0 };
__attribute__((unused))
static GLfloat floorBorder=4.0;
const float DISTANCE_BETWEEN_INODES=1.5;


#endif /* INCLUDES_CONSTANTS_H_ */
