#include "file_structures.h"
#include <iostream>
#include <boost/filesystem.hpp>
#include "support_functions.h"
#include "graphics.h"

int main(int argc, char* argv[])
{
	//initate current folder
	boost::filesystem::path cwd = boost::filesystem::current_path();
	Directory* main=new Directory(cwd);
	main->init_cwd();
	__attribute__((unused))
	Renderer* renderer=new Renderer(main);
	delete renderer;
	exit(0);

  return 0;
}
