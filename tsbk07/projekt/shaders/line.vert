#version 150

in  vec3 in_Position;

uniform mat4 projMatrix;
uniform mat4 cameraMatrix;

void main(void)
{
gl_Position = projMatrix* cameraMatrix * vec4(in_Position, 1.0);
}
