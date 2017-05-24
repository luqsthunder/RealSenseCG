#version 450 core
in vec2 texCoord;
in vec3 fragColorToVert;

uniform usampler2D uTexture;

out vec4 color;

void main()
{
  uvec4 pixel = texture(uTexture, texCoord);
  float singlepxval =  pixel.x / 606.f;
                               //5535.f;
                               //32767.f;
                               //52428.f;
  color = vec4(singlepxval, singlepxval, singlepxval, 1);
}
