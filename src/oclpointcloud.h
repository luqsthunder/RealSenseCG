#ifndef __OCL_POINT_CLOUD_H__
#define __OCL_POINT_CLOUD_H__

#define CL_VERSION_1_2 1

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

  void update(const std::vector<uint16_t>& depthImg, rscg::CameraDevice& cam);
  void draw(unsigned shader, const glm::mat4 &p) const;

private:
  cl_platform_id    cpPlatform;
  cl_context        clGPUContext;
  cl_device_id      *cdDevices;
  cl_uint           uiDevCount;
  cl_command_queue  cqCommandQueue;
  cl_kernel         pointCloudKernel, finiteDifferenceKernel;
  cl_mem            differentialImgInput, differentialImgOutput;
  cl_program        pointCloudProgramCL, finiteDifferenceProgram;

  unsigned _width, _height;
  unsigned textureGLID, quadVao, quadVbo, quadEbo;

  std::vector<unsigned> finiteDifferenceImgCPU;

  std::vector<float> vertices;
};

}

#endif