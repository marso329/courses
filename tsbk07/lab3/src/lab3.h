//Structure to hold data for every model in world
struct objects_in_world{
	Model* _model;
	mat4* _rotation;
	mat4* _translation;
	mat4* _total;

} oiw;

struct list_node{
	oiw*;
	list_node* next;

};


