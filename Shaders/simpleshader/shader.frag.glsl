#version 330 core

in vec2 texCoord;

//uniform Sample2D uTexture;

out vec4 color;

void main()
{
  color = vec4(0.8, 0.0, 1.0, 1.0);//texture(uTexture, texCoord);
}