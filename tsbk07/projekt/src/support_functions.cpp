/*
 * support_functions.cpp
 *
 *  Created on: Apr 9, 2016
 *      Author: martin
 */
#include "support_functions.h"


std::pair<int,int> get_best_rectangle(int in){
	if (in<1){
		throw std::invalid_argument("Cant create a rectangle of zero objects");
	}

	int x,y;
	x=int(ceil(sqrt(in)));
	y=ceil((float)in/(float)x);

	return std::make_pair(x,y);
}

mat4* create_matrix(mat4 matrix) {
	mat4 *temp = (mat4*) malloc(sizeof(mat4));
	memcpy(temp->m, &matrix.m, sizeof(GLfloat) * 16);
	return temp;
}

std::map<std::string,std::string>* names_in_directory(boost::filesystem::path cwd){
	std::map<std::string,std::string>* mapper= new std::map<std::string,std::string>();
	std::string str;
	for (boost::filesystem::directory_iterator itr(cwd);
			itr != boost::filesystem::directory_iterator(); ++itr) {
		if (is_regular_file(itr->status())) {
			str=itr->path().filename().native();
			boost::erase_all(str, itr->path().extension().native());
			(*mapper)[str]=itr->path().native();
		}
	}
	return mapper;

}

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

