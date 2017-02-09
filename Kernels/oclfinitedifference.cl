__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                               CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

__kernel void finitedifference(__read_only image2d_t input, 
                               __global float *output, int width)
{
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float leftDepth = 1.0;
  float rightDepth = 2.0;

  float sdasd=4;
  float gfdg=sdasd+leftDepth*0.5+2;
  float aaaa = cos(sdasd);
  float bbbb = sin(gfdg);

  output[pos.x + (pos.y * width)] = (rightDepth / 65535.f) - (leftDepth/ 65535.f);
}
