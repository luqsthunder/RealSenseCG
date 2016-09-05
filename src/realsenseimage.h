#ifndef __RSCG_COLORIMAGE_H
#define __RSCG_COLORIMAGE_H

#include <glbinding/gl/types.h>

#include <librealsense/rs.hpp>
#include <vector>
#include <set>

namespace rscg
{

class RealSenseImage
{
public:
  RealSenseImage(unsigned w = 0, unsigned h = 0);
  RealSenseImage(const rs::device& device , unsigned w, unsigned h);

  /**
   * @param data pointer to data
   */
  void update(const rs::device& device);

  void draw() const;

  /**
   * @return a array 2D with sizes of image
   */
  std::vector<unsigned> size();

protected:
  std::vector<float> _buffer;
  std::set<unsigned> _calibrateTest;
  unsigned _width, _height;

  gl::GLuint _texture;
};

}

#endif //RSCG_COLORIMAGE_H
