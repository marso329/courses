#version 150

in vec3 Normal;
in vec2 texCoord;
in vec3 position;


uniform vec3 lightSourcesDirPosArr[4];
uniform vec3 lightSourcesColorArr[4];
uniform float specularExponent[4];
uniform bool isDirectional[4];
uniform vec3 cameraPosition;
uniform sampler2D tex;

out vec4 outColor;


vec3 light_direction;
vec3 normal;
vec3 light_reflection;
vec3 looking_vector;
vec4 temp;
vec4 temp1;
float ks=0.45;
vec3 color=vec3(0.1,0.1,0.1);
uniform sampler2D texUnit,texUnit2;

void main(void)
{
		for (int i = 0; i < 4; i++) {
			if (isDirectional[i]) {
				light_direction = lightSourcesDirPosArr[i];
			} else {
				light_direction = normalize(
						position - lightSourcesDirPosArr[i]);
			}
			normal = normalize(Normal);
			light_reflection = reflect(light_direction, normal);
			looking_vector = normalize(cameraPosition - position);
			color += ks * lightSourcesColorArr[i]
					* pow(max(0, dot(light_reflection, looking_vector)),
							specularExponent[i]);
		}


			temp = texture(texUnit, texCoord);
			temp1=texture(texUnit2,texCoord);
			color *= 0.7;
			if(position[1]>2.0){
			color += vec3(temp1[0] * 0.3, temp1[1] * 0.3, temp1[2] * 0.3);
			}
			else{
			color += vec3(temp[0] * 0.3, temp[1] * 0.3, temp[2] * 0.3);
			}
		outColor = vec4(color, 1);
}
