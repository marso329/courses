#version 150


in vec3 Normal;
in vec2 outTexCoord;
out vec4 out_Color;

uniform sampler2D texUnit;
void main(void)
{
const vec3 light = vec3(0.0, 1.0, 1.0);
out_Color = texture(texUnit, outTexCoord)*dot(light, normalize(Normal));

}

