#include "realsenseimage.h"

#include <glbinding/gl/gl.h>

#include <iostream>

using namespace rscg;

inline static std::vector<unsigned>
depthToTexture(const uint16_t* depthImage, unsigned w, unsigned h)
{
  std::vector<unsigned> texture(w * h * 3);

  for(size_t y = 0; y < h; ++y)
  {
    for(size_t x = 0; x < w; ++x, ++depthImage)
    {
      //could be texture[args] = texture[args + 1] = *depthImage++
      texture[x + (y * (w + 3))]     = *depthImage;
      texture[x + (y * (w + 3)) + 1] = *depthImage;
      texture[x + (y * (w + 3)) + 2] = *depthImage;
    }
  }

  return texture;
}

RealSenseImage::RealSenseImage(unsigned int w, unsigned int h) : _width(w),
                                                                 _height(h)
{
  gl::glGenTextures(1, &_texture);
}

RealSenseImage::RealSenseImage(const rs::device &device, unsigned w,
                               unsigned h)  : RealSenseImage(w, h)
{
  update(device);
}

void RealSenseImage::draw() const
{

}

void
RealSenseImage::update(const rs::device &device)
{
  using namespace gl;

  std::vector<unsigned> rgb;

  glBindTexture(gl::GL_TEXTURE_2D, _texture);

  const uint16_t *data =
    reinterpret_cast<const uint16_t*>(device.get_frame_data(rs::stream::depth));

  rs::format format;
  switch(format)
  {
    case rs::format::any:
      throw std::runtime_error("not a valid format");
    case rs::format::z16:
    //case rs::format::disparity16:
      rgb = depthToTexture(data, _width, _height);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB,
                   GL_UNSIGNED_BYTE, rgb.data());
      break;
    case rs::format::xyz32f:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB,
                   GL_FLOAT, data);
      break;
    // Display YUYV by showing the luminance channel and packing chrominance
    // into ignored alpha channel
    case rs::format::yuyv:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0,
                   GL_LUMINANCE_ALPHA, GL_UNSIGNED_BYTE, data);
      break;
    // Display both RGB and BGR by interpreting them RGB, to show the flipped
    // byte ordering. Obviously, GL_BGR could be used on OpenGL 1.2+
    case rs::format::rgb8: case rs::format::bgr8:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB,
                   GL_UNSIGNED_BYTE, data);
      break;
    // Display both RGBA and BGRA by interpreting them RGBA, to show the
    // flipped byte ordering. Obviously, GL_BGRA could be used on OpenGL 1.2+
    case rs::format::rgba8: case rs::format::bgra8:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA,
                   GL_UNSIGNED_BYTE, data);
      break;
    case rs::format::y8:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_LUMINANCE,
                   GL_UNSIGNED_BYTE, data);
      break;
    case rs::format::y16:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_LUMINANCE,
                  GL_UNSIGNED_SHORT, data);
      break;
//    case rs::format::raw10:
//      // Visualize Raw10 by performing a naive downsample. Each 2x2 block
//      // contains one red pixel, two green pixels, and one blue pixel,
//      // so combine them into a single RGB triple.
//      rgb.clear(); rgb.resize(_width/2 * _height/2 * 3);
//      std::cout << "raw 10 lol" << std::endl;
//      auto out = rgb.data();
//      auto in0 = reinterpret_cast<const uint8_t *>(data);
//      decltype(in0) in1 = in0 + _width*5/4;
//      for(int y=0; y<_height; y+=2)
//      {
//        for(int x=0; x<_width; x+=4)
//        {
//          // RGRG -> RGB RGB
//          *out++ = in0[0]; *out++ = (in0[1] + in1[0]) / 2; *out++ = in1[1];
//          // GBGB
//          *out++ = in0[2]; *out++ = (in0[3] + in1[2]) / 2; *out++ = in1[3];
//          in0 += 5; in1 += 5;
//        }
//        in0 = in1; in1 += _width*5/4;
//      }
//      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width/2, _height/2, 0, GL_RGB,
//                   GL_UNSIGNED_BYTE, rgb.data());
//      break;
    default:
    std::cout << "default" << std::endl;
      break;
  }
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glBindTexture(GL_TEXTURE_2D, 0);
}

std::vector<unsigned>
RealSenseImage::size()
{
  return {_width, _height};
}