#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>

#include <opencv2/opencv.hpp>

#include <SFML/Network.hpp>

#include <boost/filesystem.hpp>

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

#include "realsenseimage.h"
#include "camera.h"
#include "shaderprogram.h"
#include "window.h"
#include "oclpointcloud.h"
#include "rect.h"
#include "graphprobs.h"

std::vector<float>
classifyImgNet(sf::TcpSocket &sock, const std::vector<int> &im,
               cv::Size imSize) {
  size_t reciLen;
  char predictBuff[50000];

  for(int y = 0; y < imSize.height; ++y) {
    if(sock.send((void *)(&im[y * imSize.width]),
       sizeof(int) * imSize.width, reciLen) != sf::Socket::Done) {
      std::cout << "error " << std::endl;
      exit(4);
    }
  }
  predictBuff[0] = '\0';

  sock.receive(predictBuff, sizeof(char) * 50000, reciLen);

  predictBuff[reciLen] = '\0';

  std::vector<float> resultVec;

  std::string str{predictBuff};

  for(size_t i = 0; i < 14; ++i) {
    float a = std::stof(str.substr(i * 9, 7));
    resultVec.push_back(a);
  }
  
  return resultVec;
} 

cv::Mat cutImage(const cv::Mat& im, const cv::Size &size);

void 
fillTexture(SDL_Texture * texture, cv::Mat const &mat) {
  IplImage * img = &(IplImage)mat;

  unsigned char * texture_data = NULL;
  int texture_pitch = 0;

  SDL_LockTexture(texture, 0, (void **)&texture_data, &texture_pitch);
  memcpy(texture_data, (void *)img->imageData,
         img->width * img->height * img->nChannels);
  SDL_UnlockTexture(texture);
}

inline void
findAndCutBoundingBoxFromImage(cv::Mat& imOut) {
  rscg::Rect<int> boundingBox = rscg::boundingSquare(imOut);

  if(boundingBox.w > 0) {
    cv::Mat boundedFrame2 = cv::Mat::zeros(cv::Size{boundingBox.w,
                                           boundingBox.h}, CV_8UC1);
    size_t y2, y1, x2, x1;
    y2 = 0;
    y1 = boundingBox.y;
    for(; y2 < boundingBox.h; ++y2, ++y1) {
      x2 = 0;
      x1 = boundingBox.x;
      for(; x2 < boundingBox.w; ++x2, ++x1) {
        uint8_t v = 0;

        v = imOut.at<uint8_t>(std::min((size_t)479, y1),
                              std::min((size_t)639, x1));
        if(x1 > 639 || y1 > 479) {
          continue;
        }

        boundedFrame2.at<uint8_t>(y2, x2) = v;
      }
    }

    cv::resize(boundedFrame2, imOut, cv::Size{boundingBox.w, boundingBox.w}, 0,
               0, CV_INTER_AREA);
  }
}


size_t
amountImgUncutClsFolder(int clsNum, const std::string &name) {
  namespace fs = boost::filesystem;
  std::string folder = "Gestures/dynamic_poses/uncut/" + name + "/P" +
                        std::to_string(clsNum) + "/";

  return (size_t) std::count_if(
    fs::directory_iterator(folder),
    fs::directory_iterator(),
    static_cast<bool(*)(const fs::path&)>(fs::is_regular_file));
}


bool 
saveSingleImage(const cv::Mat &imCDepth, const cv::Mat &im100, 
                const cv::Mat &imNormTotal, const cv::Mat &imModDepth,
                int imNum, int clsNum) {
  namespace fs = boost::filesystem;

  std::string folder = "Gestures/dynamic_poses/uncut/complete_depth/P" +
                        std::to_string(clsNum) + "/";
  if(!fs::is_directory(folder)) {
    fs::path p(folder);
    fs::create_directory(p);
  }


  cv::FileStorage f(folder + "im" + std::to_string(imNum) + ".xml", 
                    cv::FileStorage::WRITE);
  f << "im" << imCDepth;
  f.release();


  folder = "Gestures/dynamic_poses/uncut/100x100/P" +
                        std::to_string(clsNum) + "/";
  if(!fs::is_directory(folder)) {
    fs::path p(folder);
    fs::create_directory(p);
  }
  cv::imwrite(folder + "im" + std::to_string(imNum) + ".png", im100);


  folder = "Gestures/dynamic_poses/uncut/norm_depth/P" +
                        std::to_string(clsNum) + "/";
  if(!fs::is_directory(folder)) {
    fs::path p(folder);
    fs::create_directory(p);
  }
  cv::imwrite(folder + "im" + std::to_string(imNum) + ".png", imNormTotal);


  folder = "Gestures/dynamic_poses/uncut/mod/P" +
            std::to_string(clsNum) + "/";
  if(!fs::is_directory(folder)) {
    fs::path p(folder);
    fs::create_directory(p);
  }
  cv::imwrite(folder + "im" + std::to_string(imNum) + ".png", imModDepth);

  return true;
}

int
main(int argc, char **argv)
{
  auto device = rscg::CameraDeviceKinect();
  bool setToStop = false;

  cv::namedWindow("frame", 0);

  std::vector<int> imToClassify;
  imToClassify.resize(50 * 50 * 3);

  uint16_t maxValueCap = 0;
  const uint16_t maxDepthRS = 1840;;
  uint8_t normValue = 0;

  sf::TcpSocket sock;
  bool connected = true;
  sf::Socket::Status status = sock.connect("127.0.0.1", 31000);
  if(status != sf::Socket::Done) {
    std::cerr << "erro on connecting" << std::endl;
    connected = false;
  }

  std::vector<float> resClass;

  bool recording = false;

  int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 1;
  int thickness = 3;
  int baseline;

  int clsNum = 1;

  bool isSavingVideos = false;

  std::vector<cv::Mat> videoDepth;
  std::vector<CameraSpacePoint> joints;
  joints.resize(60 * 20);
  for(int i = 0; i < 60 * 20; ++i) {
    videoDepth.push_back(cv::Mat{424, 512, CV_16UC1, cv::Scalar(0)});
  }

  unsigned cont = 0;

  cv::VideoWriter videoWriter;

  while(!setToStop || recording) {
    device.fetchDepthFrame();
    device.fetchSkeleton();
  
    char reskey = (char)cv::waitKey(60);
    if(reskey == 'q') { 
      setToStop = true; 
    }
    else if(reskey == ' ') {
      recording = !recording;
      if(recording) {
        bool res = videoWriter.open("testVideo.avi", 
                                    CV_FOURCC('M', 'J', 'P', 'G'), 15,
                                    {512, 424}, true);
        if(!res) {
          std::cout << "cannot create video writer \n";
        }
      }
    }
   
    if(recording) {
      videoDepth[cont] = device.getDepthFrame1Chanels().clone();
      videoWriter << device.getDepthFrame1Chanels();
      ++cont;
    }

    device.renderSkeletonJointsToDepth();
    cv::imshow("frame", device.getDepthFrame3Chanels());
  }

  for(int i = 0; i < cont; ++i) {
    cv::imwrite("img_depth_" + std::to_string(i) + ".ppm", videoDepth[i]);
  }

  videoWriter.release();
  cv::destroyAllWindows();

  if(connected)
    sock.disconnect();

  return 0;
}
