__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                               CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void pointcloud(__read_only image2d_t input, __global float4 *glpos, 
                         float2 focal, float2 pp, int2 imgsize)
{
  const int2 pos = {get_global_id(0), get_global_id(1)};
  float4 vert = glpos[(pos.x) + (pos.y * imgsize.x)];

  int pxvalue = read_imageui(input, sampler, pos).x;
  float yd = ((float)pxvalue)  * (((float) pos.y) - pp.y) / focal.y;
  float xd = ((float)pxvalue)  * (((float) pos.x) - pp.x) / focal.x;;

  vert.x = xd;
  vert.y = yd;
  vert.z = pxvalue;
  vert.w = 1;
    
  glpos[(pos.x) + (pos.y * imgsize.x)] = vert;
}
