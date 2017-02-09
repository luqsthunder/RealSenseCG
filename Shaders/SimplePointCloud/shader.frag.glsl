#version 450 core

in vec4 colorToFrag;

out vec4 color;

void main()
{
  color = vec4(.0, colorToFrag.z / 5535.f , .0, 1.0);
}
