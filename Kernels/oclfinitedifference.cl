__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | 
                               CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;

__kernel void finitedifference(__constant uint *input, 
                               __global uint *output, int width)
{
}
