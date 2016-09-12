#version 330 core

layout(location = 0)in vec3 aVertices;

out vec3 colorToFrag;

void main()
{
  gl_Position = vec4(aVertices, 1.0);
  colorToFrag = aVertices;
}
