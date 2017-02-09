#include "realsenseimage.h"

#include <glbinding/gl/gl.h>

#include <iostream>

#include <cmath>

#include <glm/gtc/type_ptr.hpp>

#include "RSCGutils.h"

using namespace rscg;
using namespace gl;

std::ostream& operator <<(std::ostream& out, const glm::vec4& v)
{
  out << "[" << v.x << " " << v.y << " " << v.z << "]";
  return out;
}

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
    for(size_t x = 0; x < width; ++x)
    {
      //could be texture[args] = texture[args + 1] = *depthImage++
      auto rawDepth = (depth_image[x + (y * width)]);
      if(rawDepth > 0)
      {
        auto depthInMeters = ((float)rawDepth * scale);

        //uint8_t depth = (uint8_t)((((depthInMeters) - 0.011f)/(2.041f - 0.011f)) * 255);
        uint8_t depth = rawDepth;
        rgb_image[(x * 3) + (y * (width * 3))]     = depth;//(uint8_t)(255 - depth);
        rgb_image[(x * 3) + (y * (width * 3)) + 1] = depth;
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
  glGenTextures(1, &_texture);
  glGenVertexArrays(1, &_vao);
  glGenBuffers(1, &_vbo);
  glGenBuffers(1, &_ebo);

  glBindTexture(GL_TEXTURE_2D, _texture);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16UI, _width, _height, 0, 
               GL_RGBA_INTEGER, GL_UNSIGNED_SHORT, NULL);
  std::cout << glGetError() << std::endl;
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glBindTexture(GL_TEXTURE_2D, 0);

  createVertices({
   // vertices and texture
   -1.f,  1.f,  0.f, 0.f, 0.f,
   -1.f, -1.f,  0.f, 0.f, 1.f,
    1.f, -1.f,  0.f, 1.f, 1.f,
    1.f,  1.f,  0.f, 1.f, 0.f
  },
  {
   // indices
   0, 1, 2,
   0, 2, 3
  }, _vbo, _ebo, _vao, GL_STATIC_DRAW);

  _vertCont = 6;
}

RealSenseImage::RealSenseImage(const std::vector<uint16_t> &img, unsigned w,
                               unsigned h)  : RealSenseImage(w, h)
{
  update(img);
}

RealSenseImage::~RealSenseImage()
{
  glDeleteTextures(1, &_texture);
  glDeleteVertexArrays(1, &_vao);
  glDeleteBuffers(1, &_vbo);
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

gl::GLuint 
RealSenseImage::texture()
{
  return _texture;
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

  /// remenber the vertex state not store buffer and store the indices
  /// so you cant unbind the gl_element_array_buffer
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);
}

void
RealSenseImage::update(const std::vector<uint16_t> &imgDepth)
{
  glBindTexture(gl::GL_TEXTURE_2D, _texture);

  //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, _width, _height, 0, GL_RGB,
  //             GL_UNSIGNED_INT, imgDepth.data());

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16UI, _width, _height, 0, GL_RGB_INTEGER,
               GL_UNSIGNED_SHORT, imgDepth.data());

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

  glBindTexture(GL_TEXTURE_2D, 0);

}

/*void
RealSenseImage::updatePointCloud(const uint16_t *depthImage,
                                 rscg::Intrinsics intrinsics, 
                                 float cameraScale)
{
  size_t pointCloudVertCont = 0;
  float xDepth, yDepth, zDepth;

  for(size_t y = 0; y < _height; ++y)
  {
    for(size_t x = 0; x < _width; ++x)
    {
      if((depthImage[x + y * _width]) != 0)
      {
        zDepth = cameraScale * ((float)depthImage[x + y * _width]);

        yDepth = (cameraScale * ((float)depthImage[x + y * _width])) *
                 (( ((float)y) - intrinsics.ppy) / intrinsics.fy);

        xDepth = (cameraScale * ((float)depthImage[x + y * _width])) *
                 (( ((float)x) - intrinsics.ppx) / intrinsics.fx);

        _pointCloud[(pointCloudVertCont * 3)]     = xDepth;
        _pointCloud[(pointCloudVertCont * 3) + 1] = yDepth;
        _pointCloud[(pointCloudVertCont * 3) + 2] = zDepth;

        pointCloudVertCont++;
      }
    }
  }

  glBindBuffer(GL_ARRAY_BUFFER, _vboPointCloud);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * pointCloudVertCont * 3,
                  _pointCloud.data());
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  _vertContPointCloud = pointCloudVertCont;
}*/

std::vector<unsigned>
RealSenseImage::size()
{
  return {_width, _height};
}
