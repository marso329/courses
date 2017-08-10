#include "file_structures.h"

/*Inode*/
void Inode::set_parent(Inode* par_in) {
	parent = par_in;
}

/*File*/
File::File(boost::filesystem::path path_in):Inode(path_in){
}

/*Directory*/

/*FileStructure*/

Directory::Directory(boost::filesystem::path path_in):Inode(path_in) {
	contains = new std::vector<Inode*>;

}

void Directory::print_content() {
	for (auto it = contains->begin(); it != contains->end(); it++) {
		(*it)->print();
	}
}

void Directory::init_subdirs(){
	for (auto it = contains->begin(); it != contains->end(); it++) {
		if ((*it)->directory()){
			(*it)->init_cwd();
		}
	}
}

void Directory::init_cwd(void) {
	Inode* temp;
	int elements=1+get_number_of_elements();
	if (elements<1){
		return;

	}
size=get_best_rectangle(elements);
std::vector<std::pair<int,int>> matrix;
for(int i=0;i<size.first;i++){
	for(int j=0;j<size.second;j++){
		matrix.push_back(std::make_pair(i,j));
	}
}
	int i=0;
	for (boost::filesystem::directory_iterator itr(path);
			itr != boost::filesystem::directory_iterator(); ++itr) {
		if (is_regular_file(itr->status())) {
			temp = new File(*itr);
			temp->set_parent(this);
			contains->push_back(temp);
		}
		if (is_directory(itr->status())) {
			temp = new Directory(*itr);
			temp->set_parent(this);
			contains->push_back(temp);
		}
		temp->x_pos=DISTANCE_BETWEEN_INODES*matrix[i].first;
		temp->y_pos=DISTANCE_BETWEEN_INODES*matrix[i].second;
		i++;
	}
	temp=new Directory(path.parent_path());
	contains->push_back(temp);
	temp->x_pos=DISTANCE_BETWEEN_INODES*matrix[i].first;
	temp->y_pos=DISTANCE_BETWEEN_INODES*matrix[i].second;
	temp->parent_bool=true;


}

int Directory::get_number_of_elements(){
	//return boost::filesystem::directory_iterator(path)-boost::filesystem::directory_iterator();
	boost::filesystem::directory_iterator begin(path), end;
	return std::count_if(begin, end,
	    [](const boost::filesystem::directory_entry & d) {
	        return is_directory(d.path()) ||is_regular_file(d.path());
	});
}
