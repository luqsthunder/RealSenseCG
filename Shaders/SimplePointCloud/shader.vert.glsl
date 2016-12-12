#version 450 core

layout(location = 0)in vec3 aVertices;

out vec3 colorToFrag;

uniform mat4 uViewProj;

void main()
{
  gl_Position = uViewProj * vec4(aVertices, 1.0);
  colorToFrag = aVertices;
}
