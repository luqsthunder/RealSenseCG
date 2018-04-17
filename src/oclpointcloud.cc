#include "oclpointcloud.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <glbinding/gl/gl.h>
#include <glbinding/ContextHandle.h>

#include <glm/gtc/type_ptr.hpp>

#include <SDL.h>
#include <SDL_syswm.h>

constexpr char* GL_SHARING_EXTENSION = "cl_khr_gl_sharing";

namespace
{

const char*
oclErrorString(cl_int error)
{
  static const char* errorString[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE",
  };

  const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

  const int index = -error;

  return (index >= 0 && index < errorCount) ? errorString[index] : "";
}

cl_int 
oclGetPlatformID(cl_platform_id* clSelectedPlatformID,
                 std::string platformName)
{
  char chBuffer[1024];
  cl_uint num_platforms;
  cl_platform_id* clPlatformIDs;
  cl_int ciErrNum;
  *clSelectedPlatformID = NULL;
  cl_uint i = 0;

  ciErrNum = clGetPlatformIDs(0, NULL, &num_platforms);
  if(ciErrNum != CL_SUCCESS)
  {
    std::string error{" Error " + std::to_string(ciErrNum) +
                      " in clGetPlatformIDs Call !!!"};
    throw std::runtime_error(error);
  }
  else
  {
    if(num_platforms == 0)
    {
      std::string error{"No OpenCL platform found!"};
      throw std::runtime_error(error);
    }
    else
    {
      // if there's a platform or more, make space for ID's
      clPlatformIDs = new cl_platform_id[num_platforms * 
                                         sizeof(cl_platform_id)];
      if(clPlatformIDs == NULL) {
        std::cerr << "Failed to allocate memory for cl_platform ID's!"
                  << std::endl;

        return -3000;
      }

      ciErrNum = clGetPlatformIDs(num_platforms, clPlatformIDs, NULL);
      for(i = 0; i < num_platforms; ++i)
      {
        ciErrNum = clGetPlatformInfo(clPlatformIDs[i], CL_PLATFORM_NAME, 1024,
                                     &chBuffer, NULL);
        if(ciErrNum == CL_SUCCESS) {
          if(strstr(chBuffer, platformName.c_str()) != NULL) {
            *clSelectedPlatformID = clPlatformIDs[i];
            break;
          }
        }
      }

      if(*clSelectedPlatformID == NULL)
        *clSelectedPlatformID = clPlatformIDs[0];

      delete[] clPlatformIDs;
    }
  }

  return CL_SUCCESS;
}

std::string
loadKernelSrc(const std::string &resourcePath)
{
  std::ifstream fileStream;
  std::string content;

  fileStream.open(resourcePath.c_str());

  if(!fileStream.good())
    throw std::invalid_argument("resource Path:" + resourcePath);

  std::string line;

  for(; !fileStream.eof();)
  {
    std::getline(fileStream, line);

    content += line;
  }

  fileStream.close();

  return content;
}


}

using namespace rscg;

OclPointCloud::OclPointCloud(unsigned width, unsigned heigth) :
  _width(width), _height(heigth)
{
  cl_int ciErrNum;

  ciErrNum = oclGetPlatformID(&cpPlatform, "AMD");
  if(ciErrNum != CL_SUCCESS)
    throw std::runtime_error(oclErrorString(ciErrNum));

  // Get the number of GPU devices available to the platform
  ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, 
                            &uiDevCount);
  // Create the device list
  cdDevices = new cl_device_id[uiDevCount];
  ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount,
                            cdDevices, NULL);

  // Get device requested on command line, if any
  unsigned int uiDeviceUsed = 0;
  unsigned int uiEndDev = uiDevCount - 1;

  bool bSharingSupported = false;

  for(unsigned int i = uiDeviceUsed; 
     (!bSharingSupported && (i <= uiEndDev)); ++i)
  {
    size_t extensionSize;
    ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL,
                               &extensionSize);

    if(extensionSize > 0)
    {
      char* extensions = (char*)malloc(extensionSize);
      ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS,
                                 extensionSize, extensions, &extensionSize);

      std::string stdDevString(extensions);
      free(extensions);

      size_t szOldPos = 0;
      // extensions string is space delimited
      size_t szSpacePos = stdDevString.find(' ', szOldPos);
      while(szSpacePos != stdDevString.npos)
      {
        if(strcmp(GL_SHARING_EXTENSION, 
           stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0)
        {
          // Device supports context sharing with OpenGL
          uiDeviceUsed = i;
          bSharingSupported = true;
          break;
        }
        do
        {
          szOldPos = szSpacePos + 1;
          szSpacePos = stdDevString.find(' ', szOldPos);
        }while(szSpacePos == szOldPos);
      }
    }
  }

  cl_context_properties props[] =
  {
    CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
    CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
    CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform,
    0
  };
  clGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, 
                                 NULL, &ciErrNum);
  // end of context initializing

  cl_command_queue_properties propsqueue = 0;
  cqCommandQueue = clCreateCommandQueue(clGPUContext, cdDevices[uiDeviceUsed],
                                        propsqueue, &ciErrNum);

  auto kernelSrc = loadKernelSrc("Kernels/oclpointcloud.cl");
  const char * charPtrSrc = kernelSrc.c_str();
  size_t kernelSrcSize = kernelSrc.size();
  pointCloudProgramCL = clCreateProgramWithSource(clGPUContext, 1, &charPtrSrc,
                                        &kernelSrcSize, &ciErrNum);
  std::cerr << oclErrorString(ciErrNum) << std::endl;

  ciErrNum = clBuildProgram(pointCloudProgramCL, 1, &cdDevices[uiDeviceUsed], 0, 0, 0);
  if(ciErrNum != CL_SUCCESS)
  {
    size_t log_size;
    clGetProgramBuildInfo(pointCloudProgramCL, cdDevices[uiDeviceUsed], 
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *logError = new char[log_size];
    // Get the log
    clGetProgramBuildInfo(pointCloudProgramCL, cdDevices[uiDeviceUsed], 
                          CL_PROGRAM_BUILD_LOG, log_size, logError, NULL);
    std::cout << logError << std::endl;
    delete[] logError;
  }

  pointCloudKernel = clCreateKernel(pointCloudProgramCL, "pointcloud", &ciErrNum);
  std::cerr << oclErrorString(ciErrNum) << std::endl;

  kernelSrc = loadKernelSrc("Kernels/oclfinitedifference.cl");
  charPtrSrc = kernelSrc.c_str();
  kernelSrcSize = kernelSrc.size();
  finiteDifferenceProgram = clCreateProgramWithSource(clGPUContext, 1, 
                                                      &charPtrSrc,
                                                      &kernelSrcSize, 
                                                      &ciErrNum);
  std::cerr << oclErrorString(ciErrNum) << std::endl;

  ciErrNum = clBuildProgram(finiteDifferenceProgram, 1, 
                            &cdDevices[uiDeviceUsed], 0, 0, 0);
  if(ciErrNum != CL_SUCCESS)
  {
    size_t log_size;
    clGetProgramBuildInfo(finiteDifferenceProgram, cdDevices[uiDeviceUsed],
                          CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    char *logError = new char[log_size];
    // Get the log
    clGetProgramBuildInfo(finiteDifferenceProgram, cdDevices[uiDeviceUsed],
                          CL_PROGRAM_BUILD_LOG, log_size, logError, NULL);
    std::cout << logError << std::endl;
    delete[] logError;
  }

  finiteDifferenceKernel = clCreateKernel(finiteDifferenceProgram, 
                                          "finitedifference", &ciErrNum);
  std::cerr << oclErrorString(ciErrNum) << std::endl;


  //end of building and creating kernels cl

  differentialImgInput = clCreateBuffer(clGPUContext, CL_MEM_READ_ONLY,
                                        sizeof(uint16_t) * _width * _height,
                                        NULL, NULL);
  differentialImgOutput = clCreateBuffer(clGPUContext, CL_MEM_WRITE_ONLY,
                                         sizeof(uint16_t) * _width * _height,
                                         NULL, NULL);

  clSetKernelArg(finiteDifferenceKernel, 0, sizeof(cl_mem),
                 differentialImgInput);

  clSetKernelArg(finiteDifferenceKernel, 1, sizeof(cl_mem),
                 differentialImgOutput);
  clSetKernelArg(finiteDifferenceKernel, 2, sizeof(cl_uint),
                 &_width);
}


OclPointCloud::~OclPointCloud()
{
  if(clGPUContext) clReleaseContext(clGPUContext);
  if(cdDevices) delete[] cdDevices;

  if(cqCommandQueue) clReleaseCommandQueue(cqCommandQueue);

  if(pointCloudProgramCL) clReleaseProgram(pointCloudProgramCL);

  using namespace gl;

  glDeleteVertexArrays(1, &quadVao);
  glDeleteBuffers(1, &quadVbo);
  glDeleteBuffers(1, &quadEbo);
}

void
OclPointCloud::update(const std::vector<uint16_t>& depthImg, 
                      rscg::CameraDevice& cam)
{
  cl_int ciErrNum = 0;

  cl_event eventCL;

  clEnqueueWriteBuffer(cqCommandQueue, differentialImgInput, true, 
                       0, sizeof(uint16_t) * _width * _height, 
                       depthImg.data(), 0, NULL, &eventCL);

}

void
OclPointCloud::draw(unsigned shader, const glm::mat4 &p) const
{
}

