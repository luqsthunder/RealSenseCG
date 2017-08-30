#include <iostream>
#include <fstream>
#include <memory>
#include <climits>
#include <vector>
#include <algorithm>

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

uint16_t maxDepth = 3000;

#define MAXLINE 500000

void
toPPM(const cv::Mat &a, uint32_t currPic);


void
toPPMDist(const cv::Mat &a, uint32_t currPic);

void
toPPMColor(const cv::Mat &a, uint32_t currPic);


void
toPPMBin(const cv::Mat &a, uint32_t currPic);

void
toPPMFDepth(const cv::Mat &a, uint32_t currPic);

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

void classifyImgNet(sf::TcpSocket &sock, const std::vector<int> &im, 
                    cv::Size imSize)
{
    int len;
    int count=0;
    char sendline[MAXLINE], recvline[MAXLINE];
    std::string ok;

    for(int y = 0; y < imSize.height; ++y)
    {
      sf::Packet pack, recv;
      pack.append((char *)im[y * imSize.width], 
                  sizeof(int) * imSize.width);

      if(sock.send(pack) != sf::Socket::Done)
      {
        std::cout << "error " << std::endl;
        exit(4);
      }

      /*sock.receive(recv);
      recv >> ok;
      std::cout << ok << std::endl;
      if(ok == "ok")
      {
        std::cout << "error " << std::endl;
        exit(4);
      }*/
    }
    sf::Packet res;
    sock.receive(res);
}


int
main(int argc, char **argv)
{
  uint16_t currMax = maxDepth, normValue = 0;
  auto device = rscg::CameraDeviceWindows();

  uint32_t picCount = 0, pics, picFps = 0;
  bool print = false, setToStop = false;

  cv::Mat frame{480, 640, CV_8UC1, cv::Scalar(0)}, distance{480, 640, CV_16UC1, cv::Scalar(0)}, 
          color{480, 640, CV_8UC3, cv::Scalar(0)}, frame2{480, 640, CV_8UC1, cv::Scalar(0)}, 
          fDepth{480, 640, CV_8UC1, cv::Scalar(0)};

  int thresh = 500;

  double min, max;
  cv::Point min_loc, max_loc;

  cv::namedWindow("frame", 0);
  cv::namedWindow("frame2", 0);
  cv::namedWindow("distance", 0);
  cv::namedWindow("color", 0);
  cv::createTrackbar("thresh", "frame", &thresh, 1000);

  std::vector<uint16_t> imDepth;
  std::vector<int> imToClassify;
  imToClassify.resize(480 * 640);

  std::vector<cv::Mat> imgsDepth, imgsDist, imgsColor, fullDepth, binimgs;

  imgsDepth.resize(1000);
  imgsColor.resize(1000);
  imgsDist.resize(1000);
  fullDepth.resize(1000);
  binimgs.resize(500);

  uint16_t value = 0;

  sf::TcpSocket sock;
  sf::Socket::Status status = sock.connect("127.0.0.1", 31000);
  if(status != sf::Socket::Done)
  {
    std::cerr << "erro on connecting" << std::endl;
    exit(4);
  }

  while(! setToStop)
  {
    device.fetchDepthFrame();
    device.fetchColorFrame();

    imDepth = device.getDepthFrame1Chanels();

    currMax = maxDepth;
    for(size_t y = 0; y < 640; ++y)
    {
      for(size_t x = 0; x < 480; ++x)
      {
        value = imDepth[x * 640 + y];
        if(maxDepth < value)
          maxDepth = value;

        normValue = (uint8_t)((255.00) * ((double)value / (double)maxDepth));
        imToClassify[x * 640 + y] = (int)value;

        frame.at<uint8_t>(x, y) = (uint8_t)(( (value < (uint16_t)thresh) 
                                              && (value > 10)) ? 255 : 0);

        frame2.at<uint8_t>(x, y) = (((value < (uint16_t)thresh)
                                     && (value > 10)) ? value : 0);
      }
    }
    cv::distanceTransform(frame, distance, CV_DIST_L2, 3, CV_32F);

    cv::normalize(distance, distance, 0, 1., cv::NORM_MINMAX);

    cv::minMaxLoc(distance, &min, &max, &min_loc, &max_loc);

    for(size_t y = 0; y < 480; ++y)
    {
      for(size_t x = 0; x < 640; ++x)
        imToClassify[x + y * 640] = (255 * distance.at<float>(y, x));
    }

    classifyImgNet(sock, imToClassify, {640, 480});

    imshow("frame", frame);
    imshow("distance", distance);
    imshow("color", color);
    imshow("frame2", frame2);

    char reskey = (char)cv::waitKey(60);
    
    if(reskey == 'q')
      setToStop = true;
  }

  return 0;
}


void
toPPMColor(const cv::Mat &a, uint32_t currPic)
{
  std::ofstream file;
  file.open("image_color" + std::to_string(currPic) +".ppm", std::ios::out);

  file << "P3\n640 480\n256\n";

  Pixel value;

  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      value = a.at<Pixel>(y, x);

      file << std::to_string(value.z) << " " << std::to_string(value.y) << " "
           << std::to_string(value.x) << "  ";
    }
    file << std::endl;
  }
}

void
toPPMDist(const cv::Mat &a, uint32_t currPic)
{
  std::ofstream file;
  file.open("image_dist" + std::to_string(currPic) + ".ppm", std::ios::out);

  file << "P3\n640 480\n256\n";

  uint8_t value;

  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      value = 255 * a.at<float>(y, x);

      file << std::to_string(value) << " " << std::to_string(value) << " "
        << std::to_string(value) << "  ";
    }
    file << std::endl;
  }
}

void
toPPM(const cv::Mat &a, uint32_t currPic)
{
  std::ofstream file;
  file.open("image_depth" + std::to_string(currPic) + ".ppm", std::ios::out);

  file << "P3\n640 480\n256\n";

  uint16_t value;
  uint8_t normValue;

  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      value = a.at<uint16_t>(y, x) ;

      normValue = (uint8_t)((255.00) * ((double)value / (double)maxDepth));

      //if(value != 0)
      //  std::cout << "lol\n";

      file << std::to_string(normValue) << " " 
           << std::to_string(normValue) << " "
           << std::to_string(normValue) << "  ";
    }
    file << std::endl;
  }
}

void
toPPMFDepth(const cv::Mat &a, uint32_t currPic)
{
  std::ofstream file;
  file.open("image_depth_full" + std::to_string(currPic) + ".ppm", std::ios::out);

  file << "P3\n640 480\n256\n";

  uint16_t value;
  uint8_t normValue;

  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      value = a.at<uint16_t>(y, x);

      normValue = (uint8_t)((255.00) * ((double)value / (double)maxDepth));

      //if(value != 0)
      //  std::cout << "lol\n";

      file << std::to_string(normValue) << " "
        << std::to_string(normValue) << " "
        << std::to_string(normValue) << "  ";
    }
    file << std::endl;
  }
}

void
toPPMBin(const cv::Mat &a, uint32_t currPic)
{
  std::ofstream file;
  file.open("image_depth_bin" + std::to_string(currPic) + ".ppm", std::ios::out);

  file << "P3\n640 480\n256\n";

  uint16_t value;
  uint8_t normValue;

  for(size_t y = 0; y < 480; ++y)
  {
    for(size_t x = 0; x < 640; ++x)
    {
      value = a.at<uint8_t>(y, x);

      normValue = (uint8_t)value;//(uint8_t)((255.00) * ((double)value / (double)maxDepth));

      //if(value != 0)
      //  std::cout << "lol\n";

      file << std::to_string(normValue) << " "
        << std::to_string(normValue) << " "
        << std::to_string(normValue) << "  ";
    }
    file << std::endl;
  }
}