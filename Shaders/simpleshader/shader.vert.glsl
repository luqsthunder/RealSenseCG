#version 330 core

layout(location = 0)in vec3 aVertice;
layout(location = 1)in vec2 atexCoord;

uniform mat4 uProjView;

out vec2 texCoord;

void main()
{
  gl_Position = vec4(aVertice, 1.0);
  texCoord = atexCoord;
}
