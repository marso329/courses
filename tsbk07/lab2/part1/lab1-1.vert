#version 150

in vec3 in_Position;
in vec3 in_Normal;
uniform mat4 rotMatrix;
out vec3 out_Normal;
in vec2 inTexCoord;
out vec2 TexCoord;
float sin_t = rotMatrix[0][2];
out float sin_out;


void main(void)
{
	gl_Position = rotMatrix*vec4(in_Position, 1.0);
out_Normal=in_Normal;
TexCoord=inTexCoord;
sin_out=sin_t;
}
