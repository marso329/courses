#version 150

in  vec3 inPosition;
in  vec3 inNormal;
in vec2 inTexCoord;

// NY
uniform mat4 projMatrix;
uniform mat4 mdlMatrix;
uniform mat4 cameraMatrix;

out vec2 texCoord;
out vec3 Normal;
out vec3 position;

mat3 normalMatrix = mat3(mdlMatrix);
vec3 transformedNormal = normalMatrix * inNormal;

void main(void)
{


	texCoord = inTexCoord;
	gl_Position = projMatrix* cameraMatrix * mdlMatrix * vec4(inPosition, 1.0);
	position=inPosition;
	Normal = transformedNormal;
}
