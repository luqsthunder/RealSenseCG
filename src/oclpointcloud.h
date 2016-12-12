#ifndef __OCL_POINT_CLOUD_H__
#define __OCL_POINT_CLOUD_H__

#include <CL/cl.h>
#include <CL/cl_gl.h>

#include <SDL_video.h>
#include <vector>

#include <glm/mat4x4.hpp>

#include "camera.h"

namespace rscg
{

class OclPointCloud
{
public:
  OclPointCloud(unsigned width, unsigned heigth);
  ~OclPointCloud();

  void update(const std::vector<uint8_t>& depthImg, rscg::CameraDevice& cam);
  void draw(unsigned shader, const glm::mat4 &p) const;

private:
  cl_platform_id    cpPlatform;
  cl_context        clGPUContext;
  cl_device_id      *cdDevices;
  cl_uint           uiDevCount;
  cl_command_queue  cqCommandQueue;
  cl_kernel         ckKernel;
  cl_mem            vboCL, depthImageCL;
  cl_program        clProgram;

  unsigned _width, _height;
  unsigned _vbo, _vao, _ebo;

  std::vector<float> vertices;
};

}

#endif