#ifndef FILE_STRUCTURES_H
#define FILE_STRUCTURES_H
#define  BOOST_FILESYSTEM_NO_DEPRECATED

//standard includes
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <math.h>
//#include "graphics.h"


//non standard includes
#include "support_functions.h"
#include "constants.h"

class Inode {
public:
	Inode(boost::filesystem::path path_in) : x_pos(0),y_pos(0),path(path_in),parent(NULL) {

	}
	void set_parent(Inode*);
	bool directory() {
		return is_directory(path);
	}
	virtual void print() {
	}
	virtual void init_cwd() {
	}
	virtual void tree(int depth = 0) {
	}
	virtual ~Inode() {
	}
	float x_pos;
	float y_pos;
	boost::filesystem::path path;
	bool parent_bool=false;
	//Cube* representation;
protected:
	Inode* parent;
private:
};

class File: public Inode {
public:
	File(boost::filesystem::path);
	void print() {
		std::cout << path.filename() << std::endl;
	}
	void tree(int depth = 0) {
		std::cout << std::string(depth, ' ');
		this->print();
	}
	virtual ~File() {
	}
	;
protected:
private:

};

class Directory: public Inode {
public:
	Directory(boost::filesystem::path);
	void init_cwd();
	void init_subdirs();
	void print_content();
	void tree(int depth = 0) {

		std::cout << std::string(depth, ' ') << path.filename() << std::endl;
		for (auto it = contains->begin(); it != contains->end(); it++) {
			(*it)->tree(depth + 4);

		}
	}
	void print() {
		std::cout << path.filename() << std::endl;
	}
	int get_number_of_elements();
	virtual ~Directory() {
		//std::cout<<"deleting contains"<<std::endl;
		for(auto it=contains->begin();it!=contains->end();it++){
			delete *it;
		}
		delete contains;
	}
	;
	std::vector<Inode*>* contains;
	std::pair<int,int> size;
protected:
private:

};
#endif
