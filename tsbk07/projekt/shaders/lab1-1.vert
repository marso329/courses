#version 150

in vec3 in_Position;
in vec3 in_Normal;
in vec2 in_TexCoord;

uniform mat4 projMatrix;
uniform mat4 mdlMatrix;
uniform mat4 lookMatrix;


out vec3 Normal;
out vec2 outTexCoord;
out vec3 position;

mat3 normalMatrix = mat3(mdlMatrix);
vec3 transformedNormal = normalMatrix * in_Normal;


void main(void)
{
	gl_Position = projMatrix*lookMatrix*mdlMatrix*vec4(in_Position, 1.0);
	Normal = transformedNormal;
	 outTexCoord = in_TexCoord;
position=in_Position;
}
