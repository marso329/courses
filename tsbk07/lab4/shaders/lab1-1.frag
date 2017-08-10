#version 150


in vec3 Normal;
in vec2 outTexCoord;
in vec3 position;
uniform bool skybox;
uniform bool multitexture;
out vec4 out_Color;

uniform vec3 lightSourcesDirPosArr[4];
uniform vec3 lightSourcesColorArr[4];
uniform float specularExponent[4];
uniform bool isDirectional[4];
uniform mat4 lookMatrix;
uniform vec3 camera_position;


vec3 light_direction;
vec3 normal;
vec3 light_reflection;
vec3 looking_vector;
vec4 temp;
vec4 temp1;
float ks=0.45;
vec3 color=vec3(0.0,0.0,0.0);
uniform sampler2D texUnit,texUnit2;
void main(void)
{
if (!skybox){
//direction vector of the light to this point

for (int i=0;i<4;i++){
if (isDirectional[i]){
light_direction=lightSourcesDirPosArr[i];
}
else{
light_direction=normalize(position-lightSourcesDirPosArr[i]);
}
//normalize the normal :P
normal = normalize(Normal);
//get the light reflection vector
light_reflection=reflect(light_direction,normal);
looking_vector=normalize(camera_position-position);
color+=ks*lightSourcesColorArr[i]*pow(max(0,dot(light_reflection,looking_vector)),specularExponent[i]);
}
if (multitexture){
temp=texture(texUnit, outTexCoord);
temp1=texture(texUnit2, outTexCoord);
color+=vec3(temp[0]*0.25+temp1[0]*0.25,temp[1]*0.25+temp1[1]*0.25,temp[2]*0.25+temp1[2]*0.25);

}
else{
temp=texture(texUnit, outTexCoord);
color*=0.5;
color+=vec3(temp[0]*0.5,temp[1]*0.5,temp[2]*0.5);
}
out_Color = vec4(color, 1);
}
else{
out_Color = texture(texUnit, outTexCoord);
}
}

