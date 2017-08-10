/*
 * support_functions.h
 *
 *  Created on: Apr 9, 2016
 *      Author: martin
 */

#ifndef INCLUDES_SUPPORT_FUNCTIONS_H_
#define INCLUDES_SUPPORT_FUNCTIONS_H_
#include <utility>
#include <math.h>
#include <stdexcept>
#include <map>
#include <boost/filesystem.hpp>
#include <string>
#include <sstream>
#include <vector>
#include <boost/algorithm/string.hpp>

#include "GL_utilities.h"
#include "loadobj.h"
#include "LoadTGA.h"
#include "MicroGlut.h"
#include "VectorUtils3.h"

//returns the best rectangle size to hold the inputs number of objects
std::pair<int,int> get_best_rectangle(int);

mat4* create_matrix(mat4 matrix) ;

//return a map where the key is the filename without extension and the data is the full path top that file
std::map<std::string,std::string>* names_in_directory(boost::filesystem::path);

std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) ;

#endif /* INCLUDES_SUPPORT_FUNCTIONS_H_ */
