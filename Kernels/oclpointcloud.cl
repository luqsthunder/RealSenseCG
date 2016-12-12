  __constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST; 

__kernel void pointcloud(__read_only image2d_t input, __global float3* glpos,
                         float2 focal, float2 pp)
{
  const int2 pos = {get_global_id(0), get_global_id(1)};
  const int2 localPost = {get_local_id(0), get_local_id(1)};
  const int2 GLOBAL_SIZE = {640, 480};


  for(size_t y = pos.y; y < GLOBAL_SIZE.y; ++y)
  {
    for(size_t x = pos.y; y < GLOBAL_SIZE.x; ++x)
    {
      int4 pxvalue = read_imagei(input, sampler, (int2)(x, y));
      if(pxvalue.x != 0)
      {
        float yd;
        float xd;
        xd = ((float)pxvalue.x) * ((float)x) - pp.x / focal.x;
        yd = ((float)pxvalue.x) * ((float)y) - pp.y / focal.y;

        glpos[(x * 3) + (y * 3 * GLOBAL_SIZE.x)] = xd;
        glpos[(x * 3) + (y * 3 * GLOBAL_SIZE.x)] = yd;
        glpos[(x * 3) + (y * 3 * GLOBAL_SIZE.x)] = pxvalue.x;
      }
      else
      {
        glpos[(x * 3) + (y * 3 * GLOBAL_SIZE.x)] = -0.5f;
        glpos[(x * 3) + (y * 3 * GLOBAL_SIZE.x)] = 0;
        glpos[(x * 3) + (y * 3 * GLOBAL_SIZE.x)] = 0;
      }
    }
  }
}
