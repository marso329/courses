#include "lab4.h"


void GenerateNormals(Model* model){
 	unsigned int x, z,x_pos,z_pos;
	for (x = 0; x < (unsigned int)model->tex_width; x++)
	{
		for (z = 0; z < (unsigned int)model->tex_height; z++)
		{
			if (x==0){
				x_pos=1;
			}
			else if (x==(unsigned int)model->tex_width-1){
				x_pos=model->tex_width-2;

			}
			else{
				x_pos=x;
			}

			if (z==0){
				z_pos=1;
			}
			else if (z==(unsigned int)model->tex_height-1){
				z_pos=model->tex_height-2;
			}
			else{
				z_pos=z;
			}
			vec3 point1=vec3(model->vertexArray[(x_pos + (z_pos-1) * model->tex_width)*3 + 0],
					model->vertexArray[(x_pos + (z_pos-1) * model->tex_width)*3 + 1],
					model->vertexArray[(x_pos + (z_pos-1) * model->tex_width)*3 + 2]);

			vec3 point3=vec3(model->vertexArray[(x_pos-1 + (z_pos+1) * model->tex_width)*3 + 0],
					model->vertexArray[(x_pos-1 + (z_pos+1) * model->tex_width)*3 + 1],
					model->vertexArray[(x_pos-1 + (z_pos+1) * model->tex_width)*3 + 2]);
			vec3 point2=vec3(model->vertexArray[(x_pos+1 + (z_pos+1) * model->tex_width)*3 + 0],
					model->vertexArray[(x_pos+1 + (z_pos+1) * model->tex_width)*3 + 1],
					model->vertexArray[(x_pos+1 + (z_pos+1) * model->tex_width)*3 + 2]);
			vec3 normal=CalcNormalVector(point1,point2,point3);

			model->normalArray[(x + z * model->tex_width)*3 + 0] = normal.x;
			model->normalArray[(x + z * model->tex_width)*3 + 1] = normal.y;
			model->normalArray[(x + z * model->tex_width)*3 + 2] = normal.z;

		}
	}


}

vec3 getNormal(unsigned int x, unsigned int z,Model* model){
	GLfloat xn,yn,zn;
	if (x<0 ||z<0||x>(unsigned int)model->tex_width||z>(unsigned int)model->tex_height){
		return vec3(0,1,0);

	}
	xn=model->normalArray[(x + z * model->tex_width)*3 + 0] ;
	yn=model->normalArray[(x + z * model->tex_width)*3 + 1] ;
	zn=model->normalArray[(x + z * model->tex_width)*3 + 2] ;
	return vec3(xn,yn,zn);
}
Model* GenerateTerrain(TextureData *tex)
{
	int vertexCount = tex->width * tex->height;
	int triangleCount = (tex->width-1) * (tex->height-1) * 2;
 	unsigned int x, z;

	GLfloat *vertexArray = (GLfloat *)malloc(sizeof(GLfloat) * 3 * vertexCount);
	GLfloat *normalArray =(GLfloat *) malloc(sizeof(GLfloat) * 3 * vertexCount);
	GLfloat *texCoordArray =(GLfloat *) malloc(sizeof(GLfloat) * 2 * vertexCount);
	GLuint *indexArray =(GLuint*) malloc(sizeof(GLuint) * triangleCount*3);

	printf("bpp %d\n", tex->bpp);
	for (x = 0; x < tex->width; x++)
		for (z = 0; z < tex->height; z++)
		{
// Vertex array. You need to scale this properly
			vertexArray[(x + z * tex->width)*3 + 0] = x / 1.0;
			vertexArray[(x + z * tex->width)*3 + 1] = tex->imageData[(x + z * tex->width) * (tex->bpp/8)] / 100.0;
			vertexArray[(x + z * tex->width)*3 + 2] = z / 1.0;
// Normal vectors. You need to calculate these.
			normalArray[(x + z * tex->width)*3 + 0] = 0.0;
			normalArray[(x + z * tex->width)*3 + 1] = 1.0;
			normalArray[(x + z * tex->width)*3 + 2] = 0.0;
// Texture coordinates. You may want to scale them.
			texCoordArray[(x + z * tex->width)*2 + 0] = x; // (float)x / tex->width;
			texCoordArray[(x + z * tex->width)*2 + 1] = z; // (float)z / tex->height;
		}
	for (x = 0; x < tex->width-1; x++)
		for (z = 0; z < tex->height-1; z++)
		{
		// Triangle 1
			indexArray[(x + z * (tex->width-1))*6 + 0] = x + z * tex->width;
			indexArray[(x + z * (tex->width-1))*6 + 1] = x + (z+1) * tex->width;
			indexArray[(x + z * (tex->width-1))*6 + 2] = x+1 + z * tex->width;
		// Triangle 2
			indexArray[(x + z * (tex->width-1))*6 + 3] = x+1 + z * tex->width;
			indexArray[(x + z * (tex->width-1))*6 + 4] = x + (z+1) * tex->width;
			indexArray[(x + z * (tex->width-1))*6 + 5] = x+1 + (z+1) * tex->width;
		}

	// End of terrain generation

	// Create Model and upload to GPU:

	Model* model = LoadDataToModel(
			vertexArray,
			normalArray,
			texCoordArray,
			NULL,
			indexArray,
			vertexCount,
			triangleCount*3);
	model->tex_width=tex->width;
	model->tex_height=tex->height;
	return model;
}
GLfloat get_height(int x,int y,Model* model){
	if (x>0 &&x<model->tex_width && y>0 && y<model->tex_height){
		return model->vertexArray[(x + y * model->tex_width)*3 + 1];
	}
	else{
		return 0.0;
	}
}
GLfloat get_height2(float x,float y,Model* model){
	unsigned int x_pos,z_pos;
	GLfloat distance=-1.0;
	GLfloat temp_distance;
	GLfloat height=0.0;
	for (x_pos = 0; x_pos < (unsigned int)model->tex_width; x_pos++)
	{
		for (z_pos = 0; z_pos < (unsigned int)model->tex_height; z_pos++)
		{
// Vertex array. You need to scale this properly
			temp_distance=sqrt(pow(model->vertexArray[(x_pos + z_pos * model->tex_width)*3 + 0]-(float)x,2)+pow(model->vertexArray[(x_pos + z_pos * model->tex_width)*3 + 2]-(float)y,2));
			if (distance==-1||temp_distance<distance){
				distance=temp_distance;
				height=model->vertexArray[(x_pos + z_pos * model->tex_width)*3 + 1];

			}
		}
	}
	return height;
}


void OnTimer(int value) {
	glutPostRedisplay();
	glutTimerFunc(20, &OnTimer, value);
}



mat4* create_matrix(mat4 matrix) {
	mat4 *temp = (mat4*) malloc(sizeof(mat4));
	memcpy(temp->m, &matrix.m, sizeof(GLfloat) * 16);
	return temp;
}

void enable_depth(ModelData*) {
	glEnable(GL_DEPTH_TEST);
}
void disable_depth(ModelData*) {
	glDisable(GL_DEPTH_TEST);
}

void mouse_motion_func(int x, int y,WorldData* worlddata) {
	int window_height = get_height();
	int window_width = get_width();
	int mouse_x = x;
	int mouse_y = y;
	if (mouse_x < 0) {
		mouse_x = 0;
	}
	if (mouse_y < 0) {
		mouse_y = 0;
	}
	if (mouse_x > window_width) {
		mouse_x = window_width;
	}
	if (mouse_y > window_height) {
		mouse_y = window_width;
	}
	float rel_x = (float) mouse_x / (float) window_width;
	float rel_y = (float) mouse_y / (float) window_height;
	if (get_mouse_button() == 1) {
		float x = cos(rel_x * 360 * PI / 180.0) * sin(rel_y * 180 * PI / 180.0);
		float z = sin(rel_x * 360 * PI / 180.0) * sin(rel_y * 180 * PI / 180.0);
		float y = cos(rel_y * 180 * PI / 180.0);
		vec3 dir_vector=ScalarMult(Normalize(vec3(x,y,z)),worlddata->sphere_radius);
		worlddata->cam=VectorAdd(worlddata->lookAtPoint,dir_vector);

		//worlddata->cam = vec3(x, y, z);

	}

}

float calculate_height(float x, float z, Model* model)
{
	if (floor(x)<0 ||floor(x)>model->tex_width){
		return 0.0;
	}
	if (floor(z)<0 ||floor(z)>model->tex_height){
			return 0.0;
		}
	int width =model->tex_width;
	GLfloat* vertexArray=model->vertexArray;
        int quad = (floor(x) + floor(z)*width)*3;
        // Chooses upper or lower triangle, 1 = upper, 0 = lower
        int upper = (((x - floor(x))+(z - floor(z))) > 1)? 1 : 0;
        Point3D corner1, corner2, corner3;
        if(upper){
                // Upper triangle
                int u = 1;
                int w = 1;
                corner1.x = vertexArray[quad + (u + w*width)*3 + 0];
                corner1.y = vertexArray[quad + (u + w*width)*3 + 1];
                corner1.z = vertexArray[quad + (u + w*width)*3 + 2];
                u = 0;
                corner2.x = vertexArray[quad + (u + w*width)*3 + 0];
                corner2.y = vertexArray[quad + (u + w*width)*3 + 1];
                corner2.z = vertexArray[quad + (u + w*width)*3 + 2];
                u = 1;
                w = 0;
                corner3.x = vertexArray[quad + (u + w*width)*3 + 0];
                corner3.y = vertexArray[quad + (u + w*width)*3 + 1];
                corner3.z = vertexArray[quad + (u + w*width)*3 + 2];
        } else {
                // Lower triangle
                int u = 0;
                int w = 0;
                corner1.x = vertexArray[quad + (u + w*width)*3 + 0];
                corner1.y = vertexArray[quad + (u + w*width)*3 + 1];
                corner1.z = vertexArray[quad + (u + w*width)*3 + 2];
                u = 1;
                corner2.x = vertexArray[quad + (u + w*width)*3 + 0];
                corner2.y = vertexArray[quad + (u + w*width)*3 + 1];
                corner2.z = vertexArray[quad + (u + w*width)*3 + 2];
                u = 0;
                w = 1;
                corner3.x = vertexArray[quad + (u + w*width)*3 + 0];
                corner3.y = vertexArray[quad + (u + w*width)*3 + 1];
                corner3.z = vertexArray[quad + (u + w*width)*3 + 2];
        }
        Point3D v1, v2, normal;
        v1= VectorSub(corner2, corner1);
        v2= VectorSub(corner3, corner1);
        normal= CrossProduct(v1,v2);
        // Plane equation = A*x + B*y + C*z + D = 0
        float A,B,C,D;
        A = normal.x;
        B = normal.y;
        C = normal.z;
        D = -A*corner1.x - B*corner1.y - C*corner1.z;
        // y = - (D + A*x + C*z) / B
        return -(D + A*x + C*z) / B;
}

