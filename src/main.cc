#include <iostream>
#include <fstream>
#include <memory>
#include <climits>
#include <vector>
#include <algorithm>

#include <cstdio>

#include <opencv2\opencv.hpp>
#include <opencv2\dnn.hpp>

#include <SFML/Network.hpp>

//#include <tiny_dnn/tiny_dnn.h>

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
#include "rect.h"
#include "graphprobs.h"

uint16_t maxDepth = 3000;

#define MAXLINE 500000


template <typename T>
T 
clamp(const T& n, const T& lower, const T& upper) 
{
  return std::max(lower, std::min(n, upper));
}

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

std::vector<float>
classifyImgNet(sf::TcpSocket &sock, const std::vector<int> &im,
               cv::Size imSize, bool &stop)
{
  //std::cout << std::boolalpha << sock.isBlocking() << std::endl;

  char okStr[5];
  size_t reciLen;
  char predictBuff[50000];

  for(int y = 0; y < imSize.height; ++y)
  {
    if(sock.send((void *)(&im[y * imSize.width]),
       sizeof(int) * imSize.width, reciLen) != sf::Socket::Done)
    {
      std::cout << "error " << std::endl;
      exit(4);
    }
  }
  predictBuff[0] = '\0';

  sock.receive(predictBuff, sizeof(char) * 50000, reciLen);

  predictBuff[reciLen] = '\0';

  std::vector<float> resultVec;

  std::string str{predictBuff};
  std::cout << str << std::endl;

  for(size_t i = 0; i < 14; ++i)
  {
    float a = std::stof(str.substr(i * 9, 7));
    resultVec.push_back(a);
  }
  
  return resultVec;
} 

int
main(int argc, char **argv)
{
  uint16_t currMax = maxDepth, normValue = 0;
  auto device = rscg::CameraDeviceWindows();
  rscg::GraphProbs graph{50};

  bool setToStop = false;

  cv::Mat frame        {480, 640, CV_8UC1,  cv::Scalar(0)}, 
          dist         {480, 640, CV_16UC1, cv::Scalar(0)},
          distWithRect {480, 640, CV_8UC3,  cv::Scalar(0)},
          color        {480, 640, CV_8UC3,  cv::Scalar(0)}, 
          frame2       {480, 640, CV_8UC1,  cv::Scalar(0)},
          fDepth       {480, 640, CV_8UC1,  cv::Scalar(0)};

  int thresh = 500;

  double min, max;
  cv::Point min_loc, max_loc;

  cv::namedWindow("frame", 0);
  cv::namedWindow("frame2", 0);
  cv::namedWindow("distance", 0);
  cv::namedWindow("color", 0);
  cv::createTrackbar("thresh", "frame", &thresh, 1000);

  std::vector<uint16_t> imDepth;
  std::vector<uint8_t> imColor;

  std::vector<int> imToClassify;
  imToClassify.resize(480 * 640);

  std::vector<cv::Mat> imgsDepth, imgsDist, imgsColor, fullDepth, binimgs;

  imgsDepth.resize(1000);
  imgsColor.resize(1000);
  imgsDist.resize(1000);
  fullDepth.resize(1000);
  binimgs.resize(500);

  uint16_t value = 0;

  /*sf::TcpSocket sock;
  sf::Socket::Status status = sock.connect("127.0.0.1", 31000);
  if(status != sf::Socket::Done)
  {
    std::cerr << "erro on connecting" << std::endl;
    exit(4);
  }*/

  std::vector<float> resClass;

  int cont = 0;
  rscg::Rect<int> boundingBox;
  std::vector<rscg::Rect<int>> boxes;

  sf::Clock clk;
  clk.restart();

  bool printing = false;

  cv::Mat boundedDist = cv::Mat::ones(cv::Size{50, 50}, CV_8UC1);

  while(! setToStop || printing)
  {
    device.fetchDepthFrame();
    device.fetchColorFrame();

    imDepth = device.getDepthFrame1Chanels();
    imColor = device.getColorFrame();
    
    toOCVColor(imColor, color);

    currMax = maxDepth;
    for(size_t y = 0; y < 640; ++y)
    {
      for(size_t x = 0; x < 480; ++x)
      {
        value = imDepth[x * 640 + y];
        if(maxDepth < value)
          maxDepth = value;

        normValue = (uint8_t)((255.00) * ((double)value / (double)maxDepth));

        frame.at<uint8_t>(x, y) = (uint8_t)(( (value < (uint16_t)thresh) 
                                              && (value > 10)) ? 255 : 0);

        frame2.at<uint8_t>(x, y) = (((value < (uint16_t)thresh)
                                     && (value > 10)) ? value : 0);
        fDepth.at<uint8_t>(x, y) = normValue;

      }
    }

    cv::morphologyEx(frame, frame, CV_MOP_CLOSE, 
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {15, 15}));

    cv::distanceTransform(frame, dist, CV_DIST_L2, 3, CV_32F);

    cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);

    cv::minMaxLoc(dist, &min, &max, &min_loc, &max_loc);

    distWithRect = dist.clone();
    boundingBox = rscg::boundingSquare(dist);
    cv::rectangle(distWithRect, {boundingBox.x, boundingBox.y , boundingBox.w,
                                 boundingBox.h}, cv::Scalar(128));

    for(size_t y = 0; y < 480; ++y)
    {
      for(size_t x = 0; x < 640; ++x)
        imToClassify[x + y * 640] = (255 * dist.at<float>(y, x));
    }

    auto cir = frame;

    if(printing > 0)
      cv::circle(cir, max_loc, 10, cv::Scalar(128));

    imshow("frame", cir);
    imshow("distance", distWithRect);
    imshow("color", color);
    imshow("frame2", boundedDist);

    char reskey = (char)cv::waitKey(60);
    
    if(reskey == 'q')
      setToStop = true;
    if(reskey == ' ')
    {
      clk.restart();
      printing = true;
    }
    
    if(printing)
    {
      boundedDist = cv::Mat::ones(cv::Size{boundingBox.h, boundingBox.w}, 
                                  CV_8UC1);
      for(size_t y = boundingBox.y, y1 = 0;
          y < cv::min(boundingBox.w + boundingBox.y, 480); ++y, ++y1)
      {
        for(size_t x = boundingBox.x, x1 = 0; 
            x < cv::min(boundingBox.h + boundingBox.x, 480); ++x, ++x1)
        {
          boundedDist.at<uint8_t>(x1, y1) = 255 * dist.at<float>(y, x);
        }
      }

      imgsColor.at(cont) = color.clone();
      imgsDist.at(cont) = boundedDist.clone();
      imgsDepth.at(cont) = frame2.clone();
      fullDepth.at(cont) = fDepth.clone();
      boxes.push_back(boundingBox);
      cont++;
    }


    if(printing && clk.getElapsedTime().asSeconds() > 1)
    {
      printing = false;
      clk.restart();
    }
  }

  //sock.disconnect();

  /*for(int i = 0; i < cont; ++i)
    cv::imwrite("color_" + std::to_string(i) + ".jpg", imgsColor.at(i));
  for(int i = 0; i < cont; ++i)
    cv::imwrite("dist_" + std::to_string(i) + ".jpg", imgsDist.at(i));
  for(int i = 0; i < cont; ++i)
    cv::imwrite("depth_" + std::to_string(i) + ".jpg", imgsDepth.at(i));
  for(int i = 0; i < cont; ++i)
    cv::imwrite("fulldepth_" + std::to_string(i) + ".jpg", fullDepth.at(i));*/

  return 0;
}
