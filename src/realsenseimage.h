#ifndef __RSCG_COLORIMAGE_H
#define __RSCG_COLORIMAGE_H

#include <glbinding/gl/types.h>

#include <glm/mat4x4.hpp>

#include <vector>
#include <set>

#include "camera.h"

namespace rscg
{

/**
  * @brief This class represent, a format for change data between camera,
  * opengl, filters and other formats.
  *
  */
class RealSenseImage
{
public:
/**
  * @param w width of image.
  * @param h heigth of image
  *
  * @brief image cannot change your own size after created.
  */
  RealSenseImage(unsigned w = 0, unsigned h = 0);

/**
  * @param device reference to camera device
  * @param w the width of image
  * @param h the heigth of image
  *
  * @brief create a image with w and h size and not change after created,
  * and create image from frame aquired from camera
  */
  RealSenseImage(rscg::CameraDevice& device, unsigned w, unsigned h);

/**
  * @brief default destructor
  */
  ~RealSenseImage();

/**
  * @param data pointer to data
  *
  * @brief
  */
  void update(rscg::CameraDevice& device);

  void draw(unsigned program) const;

  void drawPointCloud(unsigned program, const glm::mat4& unfvaleu) const;

/**
  * @return a array 2D with sizes of image
  */
  std::vector<unsigned> size();

protected:
  void createVertices(const std::vector<float> &vertsAndTex,
                      const std::vector<unsigned> &indices,
                      unsigned &vbo, unsigned &ebo, unsigned &vao,
                      gl::GLenum bufferType);

  void updatePointCloud(const uint16_t *depthImage, 
                        rscg::Intrinsics intrinsics,
                        float cameraScale);

  std::vector<uint8_t>  _rgb;

  std::vector<float> _pointCloud;

  unsigned _width, _height;
  size_t _vertCont, _vertContPointCloud;

  unsigned _texture, _vbo, _vao, _ebo, _vboPointCloud, _vaoPointCloud,
           _eboPointCloud;
};

}

#endif //RSCG_COLORIMAGE_H
