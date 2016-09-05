#version 330 core

in vec2 texCoord;

uniform Sample2D uTexture;

out vec4 color;

void main()
{
  color = texture(uTexture, texoCoord);
}