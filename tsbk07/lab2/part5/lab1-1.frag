#version 150

in vec3 out_Normal;
//out vec3 out_Color1;
out vec4 out_Color;
in vec2 TexCoord;
in float sin_out;

uniform sampler2D texUnit;
void main(void)
{
//out_Color1 = out_Normal;
//out_Color = vec4(TexCoord.s, TexCoord.t, abs(sin_out), 1.0);
const vec3 light = vec3(0.58, 0.58, 0.58);
out_Color = texture(texUnit, TexCoord)*dot(light, normalize(out_Normal));
}

