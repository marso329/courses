/*
 * graphics.cpp
 *
 *  Created on: Apr 9, 2016
 *      Author: martin
 */
#include "graphics.h"

static Renderer* renderer;

void display_callback() {
	renderer->display();
}

void timer_callback(int it) {
	renderer->timer(it);
}

void keyboard_callback(unsigned int key, int x, int y, int state) {
	renderer->keyboard(key, x, y, state);
}

void mouse_callback(int x, int y) {
	renderer->mouse(x, y);
}
void mouse_press_callback(int button, int state, int x, int y) {
	renderer->mouse_press(button, state, x, y);
}

//working in degrees instead of radians
namespace MATH {
#define PI 3.14159265

double cos(double rad) {
	return std::cos(rad * PI / 180.0);
}

double sin(double rad) {
	return std::sin(rad * PI / 180.0);
}

double acos(double x) {
	return std::acos(x) * 180.0 / PI;
}

double asin(double x) {
	return std::asin(x) * 180.0 / PI;
}

static double atan(double z) {
	return std::atan(z) * 180.0 / PI;
}

static double degree_to_rad(double deg) {
	return deg / 180 * PI;
}
__attribute__((unused))
static double rad_to_degree(double rad) {
	return rad / PI * 180;
}

//return degree on unit cirle from x and y value
double degree(double x, double y) {
	double degree = MATH::atan(y / x);
	if (x > 0 && y > 0) {
		return degree;
	} else if ((x < 0 && y > 0) || (x < 0 && y < 0)) {
		return degree + 180;
	} else {
		return degree + 360;
	}
}

double distance(vec3 x, vec3 y) {
	return std::sqrt(
			std::pow(x.x - y.x, 2) + std::pow(x.y - y.y, 2)
					+ std::pow(x.z - y.z, 2));

}

}

Cube::~Cube() {
	//delete model;
	for (auto it = transformations.begin(); it != transformations.end(); it++) {
		delete (*it);
	}
	transformations.clear();
	if (letters != NULL) {
		for (auto it = letters->begin(); it != letters->end(); it++) {
			delete *it;
		}
		letters->clear();
		delete letters;
	}

}
Letter::~Letter() {
	for (auto it = transformations.begin(); it != transformations.end(); it++) {
		delete (*it);
	}
	transformations.clear();
}

void Renderer::mouse(int x, int y) {
	if (!button_released) {
		vec3 looking_dir = Normalize(VectorSub(camera, looking));
		vec3 camera_dir = Normalize(CrossProduct(looking_dir, up));
		looking.x = looking.x
				- (x - mouse_x) * DISTANCE_TO_MOVE_MOUSE * camera_dir.x;
		looking.y = looking.y
				+ (y - mouse_y) * DISTANCE_TO_MOVE_MOUSE * camera_dir.y;
		looking.z = looking.z
				- (x - mouse_x) * DISTANCE_TO_MOVE_MOUSE * camera_dir.z;
	}
	mouse_x = x;
	mouse_y = y;
}

void Renderer::mouse_press(int button, int state, int x, int y) {
	clock_t t1 = clock();
	if (time_button + TIME_FOR_DOUBLECLICK > t1 && state == 0) {
		pick(x, y);
	}

	if (state == 1) {
		time_button = clock();
	}
	button_released = state;
	mouse_x = x;
	mouse_y = y;
}

void Renderer::pick(int x, int y) {
	GLint viewport[16];
	glGetIntegerv(GL_VIEWPORT, viewport);
	GLdouble width, height;
	width = viewport[2];
	height = viewport[3];
	GLfloat x_norm = (2.0 * (double) x) / width - 1.0;
	GLfloat y_norm = 1.0 - (2 * (double) y) / height;
	vec4 ray_clip = vec4(x_norm, y_norm, -1.0, 1.0);
	mat4 proj_inverse = InvertMat4(frustum_matrix);
	vec4 ray_eye = MultVec4(proj_inverse, ray_clip);
	ray_eye = vec4(ray_eye.x, ray_eye.y, -1.0, 0.0);
	mat4 view_inverse = InvertMat4(lookAtv(camera, looking, up));
	ray_eye = MultVec4(view_inverse, ray_eye);
	vec3 ray_wor = vec3(ray_eye.x, ray_eye.y, ray_eye.z);
	ray_wor = Normalize(ray_wor);
	create_line(vec3(camera.x, camera.z, camera.y - 0.5),
			vec3(camera.x + 10.0 * ray_wor.x, camera.z + 10.0 * ray_wor.z,
					camera.y + 10.0 * ray_wor.y));
	vec3 start_point = vec3(camera.x, camera.z, camera.y);
	vec3 end_point = vec3(camera.x + ray_wor.x, camera.z + ray_wor.z,
			camera.y + ray_wor.y);

	Cube* temp;
	std::vector<Cube*> temp_vector;
	for (auto it = elements->begin(); it != elements->end(); it++) {
		temp = *it;
		if (temp->represent != NULL
				&& (dynamic_cast<File*>((*it)->represent) != NULL
						|| dynamic_cast<Directory*>((*it)->represent) != NULL)) {
			double x_center = temp->x_pos;
			double y_center = temp->y_pos;
			double z_center = 1.0;
			double radius = INODE_SPHERE_RADIUS;
			double a = pow(end_point.x - start_point.x, 2)
					+ pow(end_point.y - start_point.y, 2)
					+ pow(end_point.z - start_point.z, 2);
			double b = 2.0
					* ((end_point.x - start_point.x)
							* (start_point.x - x_center)
							+ (end_point.y - start_point.y)
									* (start_point.y - y_center)
							+ (end_point.z - start_point.z)
									* (start_point.z - z_center));
			double c = pow(start_point.x - x_center, 2)
					+ pow(start_point.y - y_center, 2)
					+ pow(start_point.z - z_center, 2) - pow(radius, 2);
			double Delta = pow(b, 2) - 4 * a * c;
			if (Delta >= 0.0) {
				temp_vector.push_back(temp);
			}
		}

	}
	bool first = true;
	__attribute__((unused))
	         Cube* closest = NULL;
	double shortest_distance = 0.0;
	for (auto it = temp_vector.begin(); it != temp_vector.end(); it++) {
		if (first) {
			shortest_distance = MATH::distance(camera,
					vec3((*it)->x_pos, 1.0, (*it)->y_pos));
			first = false;
			closest = *it;
			continue;
		}
		if (MATH::distance(camera, vec3((*it)->x_pos, 1.0, (*it)->y_pos))
				< shortest_distance) {
			shortest_distance = MATH::distance(camera,
					vec3((*it)->x_pos, 1.0, (*it)->y_pos));
			closest = *it;
		}
	}
	if (closest != NULL) {
		selected = closest;
		closest->program = loadShaders("shaders/inode.vert",
				"shaders/inode_selected.frag");
		closest->shader_changed = true;
	} else {
		selected = NULL;
	}

}

Renderer::Renderer(Directory* cwd) :
		cwd(cwd) {
	renderer = this;
	elements = new std::vector<Cube*>;
	glutInit();
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH);
	glutInitContextVersion(3, 2);
	glutInitWindowSize(600, 600);
	glutCreateWindow((char*) "TMBTF 3d file manager");
	glutDisplayFunc(display_callback);
	glutPassiveMotionFunc(mouse_callback);
	glutMouseFunc(mouse_press_callback);
	init();
	glEnable(GL_LINE_SMOOTH);
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glLineWidth(1.5);
	glutTimerFunc(20, timer_callback, 0);
	glutKeyboardFunc(keyboard_callback);
	char alpha[] =
			"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijlkmnopqrstuvwxyz0123456789 .-";
	std::vector<char> vec(alpha, alpha + sizeof(alpha) - 1);

	printable = vec;
	glutMainLoop();
}

void Renderer::init_new_directory() {
	for (auto it = elements->begin(); it != elements->end(); it++) {
		delete *it;
	}
	elements->clear();
	cwd = new_directory;
	cwd->init_cwd();
	new_directory = NULL;
	change_directory = false;
	selected = NULL;
	subinit();
	data_string = "Entered: " + cwd->path.filename().native();

}

void Renderer::delete_model(Model* model) {
	if (model != NULL) {

		if (model->vertexArray != NULL) {
			delete model->vertexArray;
		}
		if (model->normalArray != NULL) {
			delete model->normalArray;
		}

		if (model->texCoordArray != NULL) {
			delete model->texCoordArray;
		}
		if (model->colorArray != NULL) {
			delete model->colorArray;
		}
		if (model->indexArray != NULL) {
			delete model->indexArray;
		}

		delete model;
	}
}

static bool deleteAll(Cube * theElement) {
	delete theElement;
	return true;
}
Renderer::~Renderer() {

	std::remove_if(elements->begin(), elements->end(), deleteAll);
	elements->clear();
	delete elements;

	delete_model(cube_model);
	delete_model(skybox_model);

	for (auto it = letters->begin(); it != letters->end(); it++) {
		delete_model((*it).second);
	}
	delete letters;

	delete cwd;
	return;

}

void Renderer::init() {
	dumpInfo();
	glewExperimental = GL_TRUE;
	glewInit();
	//clear the error which glewinit() throws,apperently it is normal
	while (glGetError() != GL_NO_ERROR) {
	}
	// GL inits
	glClearColor(0.2, 0.2, 0.5, 0);
	glEnable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	printError("GL inits");
	cube_model = LoadModelPlus((char*) "resources/cubeplus.obj");
	camera = vec3(0, 2, 0);
	looking = vec3(2, 2, 0);
	frustum_matrix = frustum(-0.1, 0.1, -0.1, 0.1, 0.2, 50.0);
	up = vec3(0, 1, 0);
	init_letters();
	create_skybox();
	create_floor();
	create_inodes();
}

void Renderer::subinit() {
	camera = vec3(0, 2, 0);
	looking = vec3(2, 2, 0);
	up = vec3(0, 1, 0);
	//init_letters();
	create_skybox();
	create_floor();
	create_inodes();
}

void Renderer::handle_promt(unsigned int key, int state) {
	//enter=36 backspace=22
	char temp_char = XkbKeycodeToKeysym(get_display(), key, 0, state);
	if (key == 22 && promt_string.size() > 0) {
		promt_string.pop_back();
		return;
	}
	if (key == 36) {
		parse_promt();
		return;
	}

	if (std::find(printable.begin(), printable.end(), temp_char)
			!= printable.end()) {
		promt_string += temp_char;
	}

}

void Renderer::list_files() {
	data_string = "";
	for (auto it = elements->begin(); it != elements->end(); it++) {
		if ((*it)->represent != NULL) {
			data_string += (*it)->represent->path.filename().native();
			data_string += " ";
		}
	}
}

void Renderer::select_file(std::string filename) {
	data_string = "";
	for (auto it = elements->begin(); it != elements->end(); it++) {
		if ((*it)->represent != NULL
				&& (*it)->represent->path.filename().native() == filename) {
			selected = *it;
			selected->program = loadShaders("shaders/inode.vert",
					"shaders/inode_selected.frag");
			selected->shader_changed = true;
			data_string = "File selected: " + filename;
			return;

		}
	}
	data_string = "No such file exists";

}

void Renderer::help() {
	data_string +=
			"Available commands: list files, select [filename.ext|filename], help, rename [filename.ext|filename] [new_filename.ext|new_filename]";

}

void Renderer::cd(std::string filename) {
	data_string = "";
	for (auto it = elements->begin(); it != elements->end(); it++) {
		if ((*it)->represent != NULL
				&& (*it)->represent->path.filename().native()
						== filename&& dynamic_cast<Directory*>((*it)->represent) != NULL) {
			change_directory = true;
			new_directory = dynamic_cast<Directory*>((*it)->represent);

			data_string = "Entering: " + filename;
			return;
		}
	}
	data_string = "No such directory exists";
}

void Renderer::rename(std::string filename, std::string new_name) {
	for (auto it = elements->begin(); it != elements->end(); it++) {
		if ((*it)->represent != NULL
				&& (*it)->represent->path.filename().native() == filename) {
			boost::filesystem::path dest(
					(*it)->represent->path.parent_path().native() + "/"
							+ new_name);
			boost::filesystem::rename((*it)->represent->path, dest);
			(*it)->represent->path = dest;
			std::string temp = (*it)->represent->path.filename().native();
			if (temp.size() > 7) {
				temp = temp.substr(0, 7);
			}
			for (auto it2 = (*it)->letters->begin();
					it2 != (*it)->letters->end(); it2++) {
				delete *it2;
			}
			(*it)->letters->clear();
			delete (*it)->letters;
			(*it)->letters = create_text(temp,
					vec3(floorBorder + 2 * (*it)->represent->x_pos,
							floorBorder + 2 * (*it)->represent->y_pos, 2.0),
					vec3(1.0, 1.0, 0));

			data_string = "Changed name of: " + filename + " to: " + new_name;
			return;
		}
	}
	data_string = "No such file exists";
}

void Renderer::delete_file(std::string filename) {
	data_string = "";
	for (auto it = elements->begin(); it != elements->end(); it++) {
		if ((*it)->represent != NULL
				&& (*it)->represent->path.filename().native() == filename) {
			if (selected == *it) {
				selected = NULL;
			}
			boost::filesystem::remove((*it)->represent->path);
			data_string = "Deleted: " + filename;
			float x_pos, y_pos;
			x_pos = (*it)->x_pos;
			y_pos = (*it)->y_pos;
			delete *it;
			elements->erase(it);
			float temp_max_pos = 0.0;
			Cube* temp_max_cube = NULL;
			for (auto it2 = elements->begin(); it2 != elements->end(); it2++) {
				if ((*it2)->represent != NULL) {
					if (temp_max_pos == 0.0
							|| (*it2)->x_pos * (*it2)->x_pos > temp_max_pos) {
						temp_max_pos = (*it2)->x_pos * (*it2)->x_pos;
						temp_max_cube = *it2;
					}

				}
			}
			if (temp_max_cube == NULL) {
				return;
			}
			for (auto it2 = temp_max_cube->letters->begin();
					it2 != temp_max_cube->letters->end(); it2++) {
				delete *it2;
			}
			temp_max_cube->letters->clear();
			delete temp_max_cube->letters;
			for (auto it2 = temp_max_cube->transformations.begin();
					it2 != temp_max_cube->transformations.end(); it2++) {
				delete *it2;

			}
			temp_max_cube->transformations.clear();

			std::string temp =
					temp_max_cube->represent->path.filename().native();
			if (temp.size() > 7) {
				temp = temp.substr(0, 7);
			}
			temp_max_cube->x_pos = x_pos;
			temp_max_cube->y_pos = y_pos;
			temp_max_cube->letters = create_text(temp, vec3(x_pos, y_pos, 2.0),
					vec3(1.0, 1.0, 0));
			temp_max_cube->transformations.push_back(
					create_matrix(T(x_pos, 1.0, y_pos)));

			return;
		}
	}
	data_string = "No such file exists";
}

void Renderer::parse_promt() {
	std::regex self_regex(".*list files.*");
	if (std::regex_match(promt_string, std::regex(".*list files.*"))) {
		list_files();
		promt_string = "";
		return;
	}
	if (std::regex_match(promt_string, std::regex(".*select.*"))) {
		std::regex rgx("\\w+[.]*\\w*");
		auto words_begin = std::sregex_iterator(promt_string.begin(),
				promt_string.end(), rgx);
		words_begin++;
		auto words_end = std::sregex_iterator();
		if (words_begin != words_end) {
			select_file((*words_begin).str());
		} else {
			data_string = "No such file exists";
		}
		promt_string = "";
		return;
	}
	if (std::regex_match(promt_string, std::regex(".*help.*"))) {
		help();
		promt_string = "";
		return;
	}
	if (std::regex_match(promt_string, std::regex(".*cd.*"))) {
		std::regex rgx("\\w+[.]*\\w*");
		auto words_begin = std::sregex_iterator(promt_string.begin(),
				promt_string.end(), rgx);
		words_begin++;
		auto words_end = std::sregex_iterator();
		if (words_begin != words_end && (*words_begin).str() == "active") {
			if (selected != NULL) {
				cd(selected->represent->path.filename().native());
			} else {
				promt_string = "";
				data_string = "No file selected";
				return;
			}

		}

		if (words_begin != words_end) {
			cd((*words_begin).str());
		} else {
			data_string = "No such directory exists";
		}
		promt_string = "";
		return;
	}

	if (std::regex_match(promt_string, std::regex(".*rename.*"))) {
		std::regex rgx("\\w+[.]*\\w*");
		auto words_begin = std::sregex_iterator(promt_string.begin(),
				promt_string.end(), rgx);
		words_begin++;
		auto words_end = std::sregex_iterator();
		if (distance(words_begin, words_end) == 2) {
			if ((*words_begin).str() == "active") {
				if (selected != NULL) {
					rename(selected->represent->path.filename().native(),
							(*++words_begin).str());
					promt_string = "";
					return;
				} else {
					promt_string = "";
					data_string = "No file selected";
					return;
				}

			}

			std::string temp_string = (*words_begin).str();
			rename(temp_string, (*++words_begin).str());
		} else {
			data_string = "No enough arguments";
		}
		promt_string = "";
		return;
	}

	if (std::regex_match(promt_string, std::regex(".*delete.*"))) {
		std::regex rgx("\\w+[.]*\\w*");
		auto words_begin = std::sregex_iterator(promt_string.begin(),
				promt_string.end(), rgx);
		words_begin++;
		auto words_end = std::sregex_iterator();
		if (words_begin != words_end && (*words_begin).str() == "active") {
			if (selected != NULL) {
				delete_file(selected->represent->path.filename().native());
				selected = NULL;
				promt_string = "";
				return;
			} else {
				promt_string = "";
				data_string = "No file selected";
				return;
			}

		}

		if (words_begin != words_end) {
			delete_file((*words_begin).str());
		} else {
			data_string = "No such file exists";
		}
		promt_string = "";
		return;
	}

	data_string = "No such command";
	promt_string = "";
}

void Renderer::draw_promt() {
	const char* cwd_string = cwd->path.native().c_str();
	int cwd_size = strlen(cwd_string);
	int window_width = get_width();
	int selected_size = 0;
	if (selected != NULL) {
		cwd_string = selected->represent->path.native().c_str();
		selected_size = strlen(cwd_string);
	}
	unsigned int max_length = window_width / 6
			- std::max(cwd_size, selected_size) - 3;
	int y_pos = 10;
	glXWaitGL();
	if (promt_string.size() > 0) {
		int start = 0;
		unsigned int end = max_length;
		while (true) {
			if (end > promt_string.size()) {
				end = promt_string.size();
			} else {
				end = start + max_length;
			}
			const char* temp_char =
					promt_string.substr(start, end - start).c_str();
			XDrawString(get_display(), get_window(), get_context(), 10, y_pos,
					temp_char, end - start);

			//XFlush(get_display());

			if (end == promt_string.size()) {
				break;
			}

			start = end + 1;
			end = start + max_length;
			y_pos += 10;
		}
	}
	y_pos += 10;
	if (data_string.size() > 0) {
		int start = 0;
		unsigned int end = max_length;
		while (true) {
			if (end > data_string.size()) {
				end = data_string.size();
			} else {
				end = start + max_length;
			}
			const char* temp_char =
					data_string.substr(start, end - start).c_str();
			//glXWaitGL();
			XDrawString(get_display(), get_window(), get_context(), 10, y_pos,
					temp_char, end - start);

			//XFlush(get_display());

			if (end == data_string.size()) {
				break;
			}

			start = end + 1;
			end = start + max_length;
			y_pos += 10;
		}
	}
	draw_cwd();
	XFlush(get_display());
}

void Renderer::draw_cwd() {
	const char* cwd_string = cwd->path.native().c_str();
	int size = strlen(cwd_string);
	int window_width = get_width();
	XDrawString(get_display(), get_window(), get_context(),
			window_width - size * 6, 10, cwd_string, size);
	//XFlush(get_display());
	if (selected != NULL) {
		cwd_string = selected->represent->path.native().c_str();
		size = strlen(cwd_string);
		XDrawString(get_display(), get_window(), get_context(),
				window_width - size * 6, 20, cwd_string, size);
		//XFlush(get_display());
	}

}

void Renderer::check_for_collision(Cube* element) {
	if (element->represent != NULL) {
		float distance = std::sqrt(
				std::pow(element->x_pos - camera.x, 2)
						+ std::pow(element->y_pos - camera.z, 2));
		if (distance < 1.5) {
			//handle folder
			if (dynamic_cast<File*>(element->represent) == NULL) {
				if (distance < INODE_SPHERE_RADIUS) {
					change_directory = true;
					new_directory =
							dynamic_cast<Directory*>(element->represent);
				}
			} else {

				double degree = MATH::degree(camera.x - element->x_pos,
						camera.z - element->y_pos);
				camera.x = element->x_pos + 1.5 * MATH::cos(degree);
				camera.z = element->y_pos + 1.5 * MATH::sin(degree);
			}
		}
	}
}

void Renderer::keyboard(unsigned int key, int x_disp, int y_disp, int state) {
	// 111=forward, 113=left, 114=right,116=back,25=w,38=a,40=d,39=s
	vec3 lookingdir = Normalize(VectorSub(looking, camera));
	vec3 strafe = Normalize(CrossProduct(lookingdir, up));
	double distance = MATH::distance(camera, looking);
	double degree = MATH::degree(lookingdir.x, lookingdir.z);
	double x = MATH::cos(degree);
	double y = MATH::sin(degree);
	if (key == 49) {
		promt_mode = !promt_mode;
	}
	if (promt_mode) {
		handle_promt(key, state);
		return;
	}
	promt_string = "";
	data_string = "";
	if (key == 111 || key == 25) {
		camera.x += lookingdir.x * DISTANCE_TO_MOVE_KEYBOARD;
		looking.x += lookingdir.x * DISTANCE_TO_MOVE_KEYBOARD;
		camera.z += lookingdir.z * DISTANCE_TO_MOVE_KEYBOARD;
		looking.z += lookingdir.z * DISTANCE_TO_MOVE_KEYBOARD;
	} else if (key == 116 || key == 39) {
		camera.x -= lookingdir.x * DISTANCE_TO_MOVE_KEYBOARD;
		looking.x -= lookingdir.x * DISTANCE_TO_MOVE_KEYBOARD;
		camera.z -= lookingdir.z * DISTANCE_TO_MOVE_KEYBOARD;
		looking.z -= lookingdir.z * DISTANCE_TO_MOVE_KEYBOARD;
	} else if (key == 113) {
		degree -= 10;
		x = MATH::cos(degree);
		y = MATH::sin(degree);
		looking.x = camera.x + x * distance;
		looking.z = camera.z + y * distance;
	} else if (key == 114) {
		degree += 10;
		x = MATH::cos(degree);
		y = MATH::sin(degree);
		looking.x = camera.x + x * distance;
		looking.z = camera.z + y * distance;
	} else if (key == 38) {
		camera.x -= strafe.x * DISTANCE_TO_MOVE_KEYBOARD;
		looking.x -= strafe.x * DISTANCE_TO_MOVE_KEYBOARD;
		camera.z -= strafe.z * DISTANCE_TO_MOVE_KEYBOARD;
		looking.z -= strafe.z * DISTANCE_TO_MOVE_KEYBOARD;
	} else if (key == 40) {
		camera.x += strafe.x * DISTANCE_TO_MOVE_KEYBOARD;
		looking.x += strafe.x * DISTANCE_TO_MOVE_KEYBOARD;
		camera.z += strafe.z * DISTANCE_TO_MOVE_KEYBOARD;
		looking.z += strafe.z * DISTANCE_TO_MOVE_KEYBOARD;
	}

}

void Renderer::create_floor() {
	//int number_of_elements=cwd->get_number_of_elements();
	std::pair<int, int> size = cwd->size;
	GLfloat size_x, size_y;
	size_x = 2 * floorBorder + 2 * (size.first - 1) * DISTANCE_BETWEEN_INODES;
	size_y = 2 * floorBorder + 2 * (size.second - 1) * DISTANCE_BETWEEN_INODES;
	Cube* floor = new Cube();
	floor->line = false;
	floor->model = cube_model;
	floor->transformations.push_back(create_matrix(S(size_x, 1.0, size_y)));
	floor->transformations.push_back(
			create_matrix(T(size_x / 2.0, 0.0, size_y / 2.0)));
	floor->program = loadShaders("shaders/ground.vert", "shaders/ground.frag");
	GLuint tex1;
	load_image_cv("resources/tile.jpg", &tex1);
	floor->texture = tex1;
	floor->texture_unit = GL_TEXTURE0;
	floor->display_function = &Renderer::display_floor;
	floor->master = this;
	elements->push_back(floor);
}

void Renderer::create_inodes() {
	for (auto it : *cwd->contains) {
		Cube* floor = new Cube();
		floor->line = false;
		floor->represent = it;
		floor->model = cube_model;
		std::string temp = (*it).path.filename().native();
		if (temp.size() > 7) {
			temp = temp.substr(0, 7);
		}
		if ((*it).parent_bool) {
			//	temp = "..";
		}
		floor->letters = create_text(temp,
				vec3(floorBorder + 2 * it->x_pos, floorBorder + 2 * it->y_pos,
						2.0), vec3(1.0, 1.0, 0));
		floor->x_pos = floorBorder + 2 * it->x_pos;
		floor->y_pos = floorBorder + 2 * it->y_pos;
		floor->transformations.push_back(
				create_matrix(
						T(floorBorder + 2 * it->x_pos, 1.0,
								floorBorder + 2 * it->y_pos)));
		floor->program = loadShaders("shaders/inode.vert",
				"shaders/inode.frag");
		GLuint tex1;
		const char* temp_string = (*it).path.native().c_str();
		load_image_cv(temp_string, &tex1);
		if (tex1 == 0) {
			load_image_tga(temp_string, &tex1);

		}
		if (tex1 == 0) {
			File* b1 = dynamic_cast<File*>(it);
			if (b1 == NULL) {
				if ((*it).parent_bool) {
					load_image_cv("resources/folder_gray.png", &tex1, -90);
				} else {
					load_image_cv("resources/folder.png", &tex1, -90);
				}
			} else {
				load_image_cv("resources/file.png", &tex1);
			}
		}
		floor->texture = tex1;
		floor->texture_unit = GL_TEXTURE0;
		floor->display_function = &Renderer::display_inodes;
		floor->master = this;
		elements->push_back(floor);
	}

}

void Renderer::load_image_cv(const char* filename, unsigned int* texture,
		int rotation) {
	glGenTextures(1, texture);
	printError("error in create_skybox1-1");
	glBindTexture(GL_TEXTURE_2D, *texture);
	printError("error in create_skybox1-9");
	cv::Mat image = cv::imread(filename);
	if (image.empty()) {
		*texture = 0;
		return;
	} else {
		cv::Point2f src_center(image.cols / 2.0F, image.rows / 2.0F);
		cv::Mat rot_mat = getRotationMatrix2D(src_center, rotation, 1.0);
		cv::Mat dst;
		warpAffine(image, dst, rot_mat, image.size());
		image.release();
		image = dst;
		cv::flip(image, image, 0);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		printError("error in create_skybox1-10");
		glTexImage2D(GL_TEXTURE_2D,     // Type of texture
				0,       // Pyramid level (for mip-mapping) - 0 is the top level
				GL_RGB,            // Internal colour format to convert to
				image.cols, // Image width  i.e. 640 for Kinect in standard mode
				image.rows, // Image height i.e. 480 for Kinect in standard mode
				0,              // Border width in pixels (can either be 1 or 0)
				GL_BGR, // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
				GL_UNSIGNED_BYTE,  // Image data type
				image.ptr());        // The actual image data itself
		dst.release();
		rot_mat.release();

		printError("error in create_skybox1-12");
	}
}

void Renderer::load_image_tga(const char* filename, unsigned int* texture) {
	LoadTGATextureSimple((char*) filename, texture);
	if (*texture == 0) {
		return;
	}

	//glGenTextures(1, texture);
	printError("error in create_skybox1-1");
	//glBindTexture(GL_TEXTURE_2D, *texture);
	printError("error in create_skybox1-9");
}

void Renderer::create_skybox() {
	Cube* floor = new Cube();
	floor->model = LoadModelPlus((char*) "resources/skydome.obj");
	skybox_model = floor->model;
	floor->line = false;
	GLuint tex1;
	load_image_cv("resources/skydome1.jpg", &tex1);
	printError("error in create_skybox1-8");
	floor->texture = tex1;
	floor->transformations.push_back(create_matrix(T(0, -0.5, 0)));
	floor->program = loadShaders("shaders/skybox.vert", "shaders/skybox.frag");
	floor->texture_unit = GL_TEXTURE0;
	floor->display_function = &Renderer::display_skybox;
	floor->master = this;
	elements->push_back(floor);
}

void Renderer::update_letters(Cube* cube_update) {
	vec3 pos = vec3(cube_update->x_pos, cube_update->y_pos, 2.0);
	vec3 dir = Normalize(VectorSub(camera, looking));
	unsigned int size = cube_update->letters->size();
	vec3 pos_2d = vec3(dir.x, dir.z, 0);
	vec3 line = Normalize(CrossProduct(vec3(0, 0, 1), pos_2d));
	vec3 start = vec3(pos.x + ((float) size / 2.0) * line.x * TEXTDISTANCE,
			pos.y + ((float) size / 2.0) * line.y * TEXTDISTANCE, pos.z);
	line.x = -line.x;
	line.y = -line.y;
	double rotation = MATH::degree_to_rad(MATH::degree(dir.z, dir.x));
	unsigned int i = 0;
	for (auto it = cube_update->letters->begin();
			it != cube_update->letters->end(); it++) {
		for (auto it1 = (*it)->transformations.begin();
				it1 != (*it)->transformations.end(); it1++) {
			delete *it1;
		}
		(*it)->transformations.clear();
		(*it)->transformations.push_back(create_matrix(Ry(rotation)));
		(*it)->transformations.push_back(
				create_matrix(S(TEXTSIZE, TEXTSIZE, TEXTSIZE)));
		(*it)->transformations.push_back(
				create_matrix(
						T(start.x + (float) i * line.x * TEXTDISTANCE, pos.z,
								start.y + (float) i * line.y * TEXTDISTANCE)));
		i++;
	}

}

void Renderer::init_letters() {
	boost::filesystem::path letter_path = cwd->path;
	letter_path += "/resources/tests";
	std::map<std::string, std::string>* data = names_in_directory(letter_path);

	letters = new std::map<std::string, Model*>();
	for (auto it = data->begin(); it != data->end(); it++) {
		(*letters)[(*it).first] = LoadModelPlus((char*) (*it).second.c_str());

	}
	delete data;

}

void Renderer::create_line(vec3 start, vec3 end) {
	Cube* floor = new Cube();
	floor->program = loadShaders("shaders/line.vert", "shaders/line.frag");
	floor->display_function = &Renderer::display_line;
	floor->master = this;

	GLuint vertexbuffer;
	glGenBuffers(1, &vertexbuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	floor->g_vertex_buffer_data[0] = start.x;
	floor->g_vertex_buffer_data[1] = start.z;
	floor->g_vertex_buffer_data[2] = start.y;
	floor->g_vertex_buffer_data[3] = end.x;
	floor->g_vertex_buffer_data[4] = end.z;
	floor->g_vertex_buffer_data[5] = end.y;

	glBufferData(GL_ARRAY_BUFFER, sizeof(floor->g_vertex_buffer_data),
			floor->g_vertex_buffer_data, GL_STATIC_DRAW);
	floor->vertexbuffer = vertexbuffer;
	floor->line = true;
	floor->time_created = clock();
	elements->push_back(floor);

}

void Renderer::display_line(Cube* element) {
	glUseProgram(element->program);
	printError("error in line0");
	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "cameraMatrix"), 1,
	GL_TRUE, lookAtv(camera, looking, up).m);
	printError("error in line1");
	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "projMatrix"), 1,
	GL_TRUE, frustum_matrix.m);
	printError("error in line2");

	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, element->vertexbuffer);
	glVertexAttribPointer(0, // attribute 0. No particular reason for 0, but must match the layout in the shader.
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*) 0            // array buffer offset
			);
	glDrawArrays(GL_LINES, 0, 2); // 2 indices for the 2 end points of 1 line

	glDisableVertexAttribArray(0);

	printError("error in line3");
}

std::vector<Letter*>* Renderer::create_text(std::string text, vec3 pos,
		vec3 dir) {
	std::vector<Letter*>* text_elements = new std::vector<Letter*>();
	unsigned int size = text.size();
	vec3 pos_2d = vec3(dir.x, dir.y, 0);
	vec3 line = Normalize(CrossProduct(vec3(0, 0, 1), pos_2d));
	vec3 start = vec3(pos.x + ((float) size / 2.0) * line.x * TEXTDISTANCE,
			pos.y + ((float) size / 2.0) * line.y * TEXTDISTANCE, pos.z);
	line.x = -line.x;
	line.y = -line.y;
	Letter* letter;
	double rotation = std::atan(dir.x / dir.y);
	for (unsigned int i = 0; i < size; i++) {
		letter = new Letter();
		letter->line = false;
		letter->model = (*letters)[text.substr(i, 1)];
		letter->transformations.push_back(create_matrix(Ry(rotation)));
		letter->transformations.push_back(
				create_matrix(S(TEXTSIZE, TEXTSIZE, TEXTSIZE)));
		letter->transformations.push_back(
				create_matrix(
						T(start.x + (float) i * line.x * TEXTDISTANCE, pos.z,
								start.y + (float) i * line.y * TEXTDISTANCE)));
		letter->program = loadShaders("shaders/inode.vert",
				"shaders/inode.frag");
		GLuint tex1;
		LoadTGATextureSimple((char*) "resources/grass.tga", &tex1);
		letter->texture = tex1;
		letter->texture_unit = GL_TEXTURE0;
		letter->display_function = &Renderer::display_letter;
		letter->master = this;
		text_elements->push_back(letter);
	}

	return text_elements;
}

void Renderer::display_skybox(Cube* element) {
	mat4 temp_matrix;
	mat4 temp_trans = lookAtv(camera, looking, up);
	memcpy(&temp_matrix.m, &temp_trans.m, sizeof(GLfloat) * 16);
	temp_matrix.m[3] = 0;
	temp_matrix.m[7] = 0;
	temp_matrix.m[11] = 0;
	temp_matrix.m[15] = 1;
	glDisable(GL_DEPTH_TEST);
	printError("error in skybox0-1");
	printError("error in skybox0");
	glUseProgram(element->program);
	glUniformMatrix4fv(glGetUniformLocation(element->program, "cameraMatrix"),
			1,
			GL_TRUE, temp_matrix.m);
	printError("error in skybox1");
	glUniform3fv(
	glGetUniformLocation(element->program, "lightSourcesDirPosArr"), 4,
			&lightSourcesDirectionsPositions[0].x);
	printError("error in skybox2");
	glUniform3fv(glGetUniformLocation(element->program, "lightSourcesColorArr"),
			4, &lightSourcesColorsArr[0].x);
	printError("error in skybox3");
	glUniform1fv(glGetUniformLocation(element->program, "specularExponent"), 4,
			specularExponent);
	printError("error in skybox4");
	glUniform1iv(glGetUniformLocation(element->program, "isDirectional"), 4,
			isDirectional);

	printError("error in skybox");
	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "projMatrix"), 1,
	GL_TRUE, frustum_matrix.m);
	glActiveTexture(element->texture_unit);
	glUniform1i(glGetUniformLocation(element->program, "tex"),
			element->texture_unit);
	temp_trans = IdentityMatrix();
	printError("display 2");

	for (auto it2 = element->transformations.begin();
			it2 != element->transformations.end(); it2++) {
		temp_trans = Mult(*(*it2), temp_trans);
	}
	glUniformMatrix4fv(glGetUniformLocation(element->program, "mdlMatrix"), 1,
	GL_TRUE, temp_trans.m);
	printError("display 3");
	glUniform3fv(glGetUniformLocation(element->program, "cameraPosition"), 1,
			&camera.x);
	glBindTexture(GL_TEXTURE_2D, element->texture);
	printError("display 4");

	DrawModel(element->model, element->program, (char*) "inPosition",
	NULL, (char*) "inTexCoord");
	glEnable(GL_DEPTH_TEST);
}

void Renderer::display_floor(Cube* element) {
	mat4 temp_trans;
	glUseProgram(element->program);
	glUniform3fv(
	glGetUniformLocation(element->program, "lightSourcesDirPosArr"), 4,
			&lightSourcesDirectionsPositions[0].x);
	glUniform3fv(glGetUniformLocation(element->program, "lightSourcesColorArr"),
			4, &lightSourcesColorsArr[0].x);
	glUniform1fv(glGetUniformLocation(element->program, "specularExponent"), 4,
			specularExponent);
	glUniform1iv(glGetUniformLocation(element->program, "isDirectional"), 4,
			isDirectional);

	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "cameraMatrix"), 1,
	GL_TRUE, lookAtv(camera, looking, up).m);
	printError("display 1");
	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "projMatrix"), 1,
	GL_TRUE, frustum_matrix.m);
	glActiveTexture(element->texture_unit);
	glUniform1i(glGetUniformLocation(element->program, "tex"),
			element->texture_unit);
	temp_trans = IdentityMatrix();
	printError("display 2");

	for (auto it2 = element->transformations.begin();
			it2 != element->transformations.end(); it2++) {
		temp_trans = Mult(*(*it2), temp_trans);
	}
	glUniformMatrix4fv(glGetUniformLocation(element->program, "mdlMatrix"), 1,
	GL_TRUE, temp_trans.m);
	printError("display 3");
	glUniform3fv(glGetUniformLocation(element->program, "cameraPosition"), 1,
			&camera.x);
	glBindTexture(GL_TEXTURE_2D, element->texture);
	printError("display 4");

	DrawModel(element->model, element->program, (char*) "inPosition",
			(char*) "inNormal", (char*) "inTexCoord");

}

void Renderer::display_inodes(Cube* element) {
	mat4 temp_trans;
	glUseProgram(element->program);
	glUniform3fv(
	glGetUniformLocation(element->program, "lightSourcesDirPosArr"), 4,
			&lightSourcesDirectionsPositions[0].x);
	glUniform3fv(glGetUniformLocation(element->program, "lightSourcesColorArr"),
			4, &lightSourcesColorsArr[0].x);
	glUniform1fv(glGetUniformLocation(element->program, "specularExponent"), 4,
			specularExponent);
	glUniform1iv(glGetUniformLocation(element->program, "isDirectional"), 4,
			isDirectional);

	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "cameraMatrix"), 1,
	GL_TRUE, lookAtv(camera, looking, up).m);
	printError("display 1");
	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "projMatrix"), 1,
	GL_TRUE, frustum_matrix.m);
	glActiveTexture(element->texture_unit);
	glUniform1i(glGetUniformLocation(element->program, "tex"),
			element->texture_unit);
	temp_trans = IdentityMatrix();
	printError("display 2");

	for (auto it2 = element->transformations.begin();
			it2 != element->transformations.end(); it2++) {
		temp_trans = Mult(*(*it2), temp_trans);
	}
	glUniformMatrix4fv(glGetUniformLocation(element->program, "mdlMatrix"), 1,
	GL_TRUE, temp_trans.m);
	printError("display 3");
	glUniform3fv(glGetUniformLocation(element->program, "cameraPosition"), 1,
			&camera.x);
	glBindTexture(GL_TEXTURE_2D, element->texture);
	printError("display 4");

	DrawModel(element->model, element->program, (char*) "inPosition",
			(char*) "inNormal", (char*) NULL);

	Cube* temp;
	update_letters(element);
	if (MATH::distance(camera, vec3(element->x_pos, 2.0, element->y_pos)) < 3.0
			|| element == selected) {

		for (auto it = element->letters->begin(); it != element->letters->end();
				it++) {
			temp = (*it);
			(temp->master->*temp->Cube::display_function)(temp);
		}
	}
}
void Renderer::display_letter(Cube* element) {
	mat4 temp_trans;
	glUseProgram(element->program);
	glUniform3fv(
	glGetUniformLocation(element->program, "lightSourcesDirPosArr"), 4,
			&lightSourcesDirectionsPositions[0].x);
	glUniform3fv(glGetUniformLocation(element->program, "lightSourcesColorArr"),
			4, &lightSourcesColorsArr[0].x);
	glUniform1fv(glGetUniformLocation(element->program, "specularExponent"), 4,
			specularExponent);
	glUniform1iv(glGetUniformLocation(element->program, "isDirectional"), 4,
			isDirectional);

	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "cameraMatrix"), 1,
	GL_TRUE, lookAtv(camera, looking, up).m);
	printError("display 1");
	glUniformMatrix4fv(
	glGetUniformLocation(element->program, "projMatrix"), 1,
	GL_TRUE, frustum_matrix.m);
	glActiveTexture(element->texture_unit);
	glUniform1i(glGetUniformLocation(element->program, "tex"),
			element->texture_unit);
	temp_trans = IdentityMatrix();
	printError("display 2");

	for (auto it2 = element->transformations.begin();
			it2 != element->transformations.end(); it2++) {
		temp_trans = Mult(*(*it2), temp_trans);
	}
	glUniformMatrix4fv(glGetUniformLocation(element->program, "mdlMatrix"), 1,
	GL_TRUE, temp_trans.m);
	printError("display 3");
	glUniform3fv(glGetUniformLocation(element->program, "cameraPosition"), 1,
			&camera.x);
	glBindTexture(GL_TEXTURE_2D, element->texture);
	printError("display 4");

	DrawModel(element->model, element->program, (char*) "inPosition",
			(char*) "inNormal", (char*) NULL);
}

bool check_for_deletion(Cube* element) {

	if (element->line
			&& element->time_created + TIME_FOR_LINES_TO_DISAPPEAR < clock()) {
		return true;
	} else {
		return false;
	}

}

void Renderer::check_for_color_change(Cube* element) {
	if (element->represent != NULL
			&& (dynamic_cast<File*>(element->represent) != NULL
					|| dynamic_cast<Directory*>(element->represent) != NULL)
			&& element->shader_changed && element != selected) {
		element->shader_changed = false;
		element->program = loadShaders("shaders/inode.vert",
				"shaders/inode.frag");

	}

}

void Renderer::display() {
	if (change_directory) {
		init_new_directory();
	}
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	Cube* temp;
	std::vector<Cube*> temp_vector;
	for (auto it = elements->begin(); it != elements->end(); it++) {

		temp = (*it);
		(temp->master->*temp->Cube::display_function)(temp);
		check_for_collision(temp);
		if (check_for_deletion(temp)) {
			temp_vector.push_back(temp);
		}
		check_for_color_change(temp);
	}
	for (auto it = temp_vector.begin(); it != temp_vector.end(); it++) {
		auto it2 = std::find(elements->begin(), elements->end(), *it);
		if (it2 != temp_vector.end()) {
			delete *it2;
			elements->erase(it2);
		}
	}
	glutSwapBuffers();
	draw_promt();
	//draw_cwd();
}

void Renderer::timer(int i) {
	glutTimerFunc(20, timer_callback, i);
	glutPostRedisplay();
}

