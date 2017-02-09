#version 450 core

layout(location = 0)in vec4 aVertices;

out vec4 colorToFrag;

uniform mat4 uViewProj;

void main()
{
  gl_Position = uViewProj * aVertices;
  colorToFrag = aVertices;
}
