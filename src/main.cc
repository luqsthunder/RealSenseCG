#include <iostream>
#include <fstream>
#include <memory>
#include <climits>
#include <vector>
#include <algorithm>

#include <opencv2\opencv.hpp>

#include <SDL.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable: 4251)
#endif

#include <glbinding/gl/gl.h>
#include <glbinding/Binding.h>

#ifdef main
#undef main
#endif

#ifndef _MSC_VER
#pragma warning(pop)
#endif


#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include "realsenseimage.h"
#include "camera.h"
#include "shaderprogram.h"
#include "window.h"
#include "oclpointcloud.h"

void
toPPM(const cv::Mat &a, bool dist = true);

void
toPPMImg(const cv::Mat &a);

void
toOCV(const std::vector<uint16_t> &in,
      cv::Mat &out)
{
  cv::Mat out1{640, 480, CV_16UC1};
  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      out1.at<uint16_t>(x, y) = in[x + y * 640];
    }
  }
  out = out1.clone();
}

typedef cv::Point3_<uint8_t> Pixel;

void
toOCVColor(const std::vector<uint8_t> &in, cv::Mat &out)
{
  for(size_t y = 0; y < 640; ++y)
  {
    for(size_t x = 0; x < 480; ++x)
    {
      out.at<Pixel>(x, y) = Pixel{in[3 * (x * 640 + y)], 
                                  in[3 * (x * 640 + y) + 1], 
                                  in[3 * (x * 640 + y) + 2]};
    }
  }
}

int
main(int argc, char **argv)
{
  auto device = rscg::CameraDeviceWindows();

  uint32_t picCount = 0;

  cv::Mat frame{480, 640, CV_8UC1}, distance{480, 640, CV_16U}, 
          color{480, 640, CV_8UC3};

  int thresh = 500;

  double min, max;
  cv::Point min_loc, max_loc;

  cv::namedWindow("frame", 0);
  cv::namedWindow("distance", 0);
  cv::namedWindow("color", 0);
  cv::createTrackbar("thresh", "frame", &thresh, 1000);

  std::vector<uint16_t> imDepth;

  std::vector<cv::Mat> imgsDepths, imgsDist, imgsColor;

  imgsDepths.resize(1000);
  imgsColor.resize(1000);
  imgsDist.resize(1000);

  for(auto &it : imgsDepths) it = cv::Mat{480, 640, CV_8UC1};
  for(auto &it : imgsDist)   it = cv::Mat{480, 640, CV_16U};
  for(auto &it : imgsColor)  it = cv::Mat{480, 640, CV_8UC3};

  int photoCount;

  for(;;)
  {
    device.fetchDepthFrame();
    device.fetchColorFrame();

    imDepth = device.getDepthFrame1Chanels();
    toOCVColor(device.getColorFrame(), color);

    for(size_t y = 0; y < 640; ++y)
    {
      for(size_t x = 0; x < 480; ++x)
      {
        frame.at<uint8_t>(x, y) = (((imDepth[x * 640 + y] < (uint16_t)thresh) && (imDepth[x * 640 + y] > 10)) ? 255 : 0);
      }
    }
    cv::distanceTransform(frame, distance, CV_DIST_L2, 3, CV_32F);

    /*cv::GaussianBlur(distance, distance, cv::Size(25, 25), 0.8, 0.8,
    cv::BORDER_DEFAULT); */

    cv::normalize(distance, distance, 0, 1., cv::NORM_MINMAX);

    cv::minMaxLoc(distance, &min, &max, &min_loc, &max_loc);

    int radius = distance.at<float>(max_loc.y, max_loc.x);

    cv::circle(frame, max_loc, radius, cv::Scalar(0, 255, 0), 2, 8, 0);

    imshow("frame", frame);
    imshow("distance", distance);
    imshow("color", color);

    if((char)cv::waitKey(5) == 'q') break;
    if((char)cv::waitKey(5) == 32) 
    {

    }
  }

  return 0;
}

int cont = 0;
int contdist = 0;

void
toPPMImg(const cv::Mat &a)
{
  std::ofstream file;
  std::string aux;
  int contAux = cont;
  file.open("imagecolor" + aux + std::to_string(contAux) + ".ppm", std::ios::out);

  uint8_t value;

  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      file << std::to_string(a.at<uint8_t>(x, y)) << " " 
           << std::to_string(a.at<uint8_t>(x, y + 1)) << " "
           << std::to_string(a.at<uint8_t>(x, y + 2)) << "  ";
    }
    file << std::endl;
  }
}

void
toPPM(const cv::Mat &a, bool dist)
{
  std::ofstream file;
  std::string aux = dist ? "" : "_distance";
  int contAux = dist ? contdist : cont;
  if(dist)
    contdist++;
  else
    cont++;
  file.open("imagedepth" + aux + std::to_string(contAux) + ".ppm", std::ios::out);

  uint16_t value;

  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      value = a.at<uint16_t>(x, y);

      file << std::to_string(value) << " " << std::to_string(value) << " "
        << std::to_string(value) << "  ";
    }
    file << std::endl;
  }
}

  /*
  void
  printToFile(const std::vector<uint16_t> &a)
  {
  std::ofstream file;
  file.open("imagedepth.txt", std::ios::out);

  uint16_t value;

  for(size_t y = 0; y < 480; ++y)
  {
  for(size_t x = 0; x < 640; ++x)
  {
  value = a[(x * 3) + (y * (640* 3))];

  file << std::to_string(value) << " " << std::to_string(value) << " "
  << std::to_string(value) << "  ";
  }
  file << std::endl;
  }
  }

  int
  main(int argc, char **argv)
  {
  using namespace gl;

  rscg::Window window{1366, 768};

  glbinding::Binding::initialize();

  auto device = rscg::CameraDeviceWindows{};

  bool running = true;
  SDL_Event event;
  rscg::RealSenseImage depthImage{640, 480};

  rscg::ShaderProgram textureProgram{"Shaders/simpleshader",
  {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}};
  rscg::ShaderProgram pointCloudProgram{"Shaders/SimplePointCloud",
  {GL_VERTEX_SHADER, GL_FRAGMENT_SHADER}};

  // rscg::OclPointCloud pointCloudOcl{640, 480};

  glm::mat4 proj, view;
  proj = glm::perspective(45.f, 1366.f/768.f, 0.01f, (float)USHRT_MAX);
  view = glm::lookAt(glm::vec3{0.f, 0.f, -1.f}, glm::vec3{0.f, 0.f, 1.f},
  glm::vec3{0.f, -1.f, 0.f});

  while(running)
  {
  auto imgDepth = device.getDepthFrame4Chanels();

  while(SDL_PollEvent(&event))
  {
  if((event.type == SDL_QUIT) || ((event.type == SDL_KEYUP) &&
  event.key.keysym.sym == SDLK_ESCAPE))
  running = false;
  if(event.type == SDL_KEYUP && event.key.keysym.sym == SDLK_SPACE)
  {
  toPPM(imgDepth);
  printToFile(imgDepth);
  }
  }

  glClearColor(0.f, 0.f, 0.f, 0.f);
  glClear(gl::GL_COLOR_BUFFER_BIT);

  device.fetchDepthFrame();
  depthImage.update(imgDepth);
  depthImage.draw(textureProgram.programID());
  //pointCloudOcl.update(imgDepth, device);
  //pointCloudOcl.draw(pointCloudProgram.programID(), proj * view);

  SDL_GL_SwapWindow(window);
  }
  return 0;
  }
  */