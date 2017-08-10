#version 150

in vec3 in_Position;
in vec3 in_Normal;
uniform mat4 rotMatrix;
out vec3 out_Normal;


void main(void)
{
	gl_Position = rotMatrix*vec4(in_Position, 1.0);
out_Normal=in_Normal;
}