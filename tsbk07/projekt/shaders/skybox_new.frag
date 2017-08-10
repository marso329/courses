#version 330

in vec3 texCoord;

out vec4 outColor;

uniform samplerCube skybox;

void main(void)
{
outColor = texture(skybox, texCoord);
}
