#include "realsenseimage.h"

#include <glbinding/gl/gl.h>

#include <iostream>

#include <cmath>

using namespace rscg;
using namespace gl;


void print(const std::vector<uint8_t > &v)
{
  for(size_t y = 0; y < 640; ++y)
  {
    for(size_t x = 0; x < 480; ++x)
    {
      std::cout << "["
                << (int)v[x + (y * 640 * 3)]     << " "
                << (int)v[x + (y * 640 * 3) + 1] << " "
                << (int)v[x + (y * 640 * 3) + 2]
                << "]";
    }
    std::cout << std::endl;
  }
}

void
depthToTexture(const uint16_t* depth_image, unsigned width, unsigned height,
               std::vector<uint8_t> &rgb_image, float scale)
{
  if(rgb_image.size() < (width * height * 3))
    rgb_image.resize(width * height * 3);

  for(size_t y = 0; y < height; ++y)
  {
    for(size_t x = 0; x < width; ++x, ++depth_image)
    {

      //could be texture[args] = texture[args + 1] = *depthImage++
      auto rawDepth = (*depth_image);// % (256 * 32);
      if(rawDepth > 0)
      {
        auto depthInMeters = ((float)rawDepth * scale);

        uint8_t depth = (uint8_t)((((depthInMeters) - 1.f)/(1.f - 0.2f)) * 256);
        rgb_image[(x * 3) + (y * (width * 3))]     = (uint8_t)(255 - depth);
        rgb_image[(x * 3) + (y * (width * 3)) + 1] = 0;
        rgb_image[(x * 3) + (y * (width * 3)) + 2] = depth;
      }
      else
      {
        rgb_image[(x * 3) + (y * (width * 3))]     = 0;
        rgb_image[(x * 3) + (y * (width * 3)) + 1] = 0;
        rgb_image[(x * 3) + (y * (width * 3)) + 2] = 0;
      }
    }
  }
}

RealSenseImage::RealSenseImage(unsigned int w, unsigned int h) : _width(w),
                                                                 _height(h)
{
  _rgb.resize(_width * _height * 3);

  glGenTextures(1, &_texture);
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);
  glGenBuffers(1, &_ebo);

  createVertices({
   // vertices and texture
   -1.f, 1.f, 0.f, 0.f, 0.f,
   -1.f, -1.f, 0.f, 0.f, 1.f,
    1.f, -1.f, 0.f, 1.f, 1.f,
    1.f, 1.f, 0.f, 1.f, 0.f
  },
  {
   // indices
   0, 1, 2,
   0, 2, 3
  }, _vbo, _ebo, _vao, GL_STATIC_DRAW);

  _vertCont = 6;

  glGenBuffers(1,& _vboPointCloud);
  glGenBuffers(1,& _eboPointCloud);
  glGenVertexArrays(1, &_vaoPointCloud);
}

RealSenseImage::RealSenseImage(const rs::device &device, unsigned w,
                               unsigned h)  : RealSenseImage(w, h)
{
  update(device);
}

RealSenseImage::~RealSenseImage()
{
  glDeleteTextures(1, &_texture);
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo);

  glDeleteVertexArrays(1, &_vaoPointCloud);
  glDeleteBuffers(1, &_vboPointCloud);
  glDeleteBuffers(1, &_eboPointCloud);
}

void
RealSenseImage::draw(unsigned program) const
{
  glUseProgram(program);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, _texture);
  glUniform1i(glGetUniformLocation(program, "uTexture"), 0);

  glBindVertexArray(_vao);
  glDrawElements(GL_TRIANGLES, (gl::GLsizei)_vertCont,
                 GL_UNSIGNED_INT, 0);

  glBindTexture(GL_TEXTURE_2D, 0);
  glBindVertexArray(0);
}

void
RealSenseImage::drawPointCloud(unsigned program) const
{
  glUseProgram(program);

  glBindVertexArray(_vaoPointCloud);

  glDrawArrays(GL_POINTS, 0, (GLsizei)256);

  //glDrawElements(GL_POINTS, (gl::GLsizei)256,
  //               GL_UNSIGNED_INT, 0);

  glBindVertexArray(0);
}

void
RealSenseImage::createVertices(const std::vector<float> &vertAndTex,
                               const std::vector<unsigned> &indices,
                               unsigned &vbo, unsigned &ebo, unsigned &vao,
                               gl::GLenum bufferType)
{
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);

  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertAndTex.size(),
               vertAndTex.data(), bufferType);

  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * indices.size(),
               indices.data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                        5 * sizeof(float), (void*) nullptr);
  glEnableVertexAttribArray(0);

  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE,
                        5 * sizeof(float), (void*) (3 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // remenber the vertex state not store buffer and store the indices
  // so you cant unbind the gl_element_array_buffer
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void
RealSenseImage::update(const rs::device &device)
{
  if(_buffer.size() == 0)
  {
    _buffer.resize(_width * _height * 3);
    glBindVertexArray(_vaoPointCloud);

    glBindBuffer(GL_ARRAY_BUFFER, _vboPointCloud);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _eboPointCloud);

    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * _buffer.size(),
                 _buffer.data(), GL_STATIC_DRAW);

    std::vector<unsigned> indices;
    for(unsigned y = 0; y < _height; ++y)
    {
      for(unsigned x = 0; x < _width; ++x)
      {
        indices.push_back(x + (y * _width));
        indices.push_back(x + ((y + 1) * _width));
        indices.push_back((x + 1) + ((y + 1) * _width));

        indices.push_back(x + (y * _width));
        indices.push_back((x + 1) + ((y + 1) * _width));
        indices.push_back((x + 1) + (y * _width));
      }
    }
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned) * indices.size(),
                 indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                          3 * sizeof(float), (void*) nullptr);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  glBindTexture(gl::GL_TEXTURE_2D, _texture);

  const uint16_t *data =
    reinterpret_cast<const uint16_t*>(device.get_frame_data(rs::stream::depth));

  rs::format format = device.get_stream_format(rs::stream::depth);
  auto sensorDepthScale = device.get_depth_scale();
  switch(format)
  {
    case rs::format::any:
      throw std::runtime_error("not a valid format");
  case rs::format::z16: case rs::format::disparity16:
    {
      depthToTexture(data, _width, _height, _rgb, sensorDepthScale);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB,
                   GL_UNSIGNED_BYTE, _rgb.data());

      auto intrinsics = device.get_stream_intrinsics(rs::stream::depth);
      updatePointCloud(data, intrinsics, device.get_depth_scale());
      break;
    }
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

void
RealSenseImage::updatePointCloud(const uint16_t *depthImage,
                                 rs::intrinsics intrinsics,  float cameraScale)
{
  size_t pointCloudVertCont = 0;
  float xDp,yDp,zDp;
  for(size_t y = 0; y < _height; ++y)
  {
    for(size_t x = 0; x < _width; ++x)
    {
      auto rawDepth = *depthImage;
      if(rawDepth != 0)
      {
        /*float normDepth = ((( ((float)rawDepth) * cameraScale)  - 1.f) /
                           (1.f - 0.2f));

        yDp = (normDepth * (float)x) / std::sqrt(std::pow(cameraFy, 2) +
                                          std::pow(y, 2));

        xDp = (normDepth * (float)y) / std::sqrt(std::pow(cameraFx, 2) +
                                          std::pow(x, 2));
        zDp = std::sqrt(std::pow(normDepth, 2) - std::pow(y, 2));*/


        auto dp = intrinsics.deproject({x, y}, rawDepth);
        
        _buffer[(pointCloudVertCont * 3)]     = dp.x;
        _buffer[(pointCloudVertCont * 3) + 1] = dp.y;
        _buffer[(pointCloudVertCont * 3) + 2] = dp.z;
      }
    }
  }
  glBindBuffer(GL_ARRAY_BUFFER, _vboPointCloud);
  glBufferData(GL_ARRAY_BUFFER, pointCloudVertCont, _buffer.data(),
               GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

std::vector<unsigned>
RealSenseImage::size()
{
  return {_width, _height};
}
