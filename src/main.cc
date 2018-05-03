#include <iostream>
#include <fstream>
#include <memory>
#include <climits>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <cstdio>
#include <climits>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

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


class ImageClassifier {
public:
  ImageClassifier();

  const std::vector<float>& classifyImage(const cv::Mat &im);
private:
  sf::Socket _socket;
};

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
                const cv::Mat &imNormTotal, int imNum, int clsNum) {
  namespace fs = boost::filesystem;

  std::string folder = "Gestures/dynamic_poses/uncut/complete_depth/P" +
                        std::to_string(clsNum) + "/";
  if(!fs::is_directory(folder)) {
    fs::path p(folder);
    fs::create_directory(p);
  }
  cv::imwrite(folder + "imCDp" + std::to_string(imNum) + ".ppm", imCDepth);


  folder = "Gestures/dynamic_poses/uncut/100x100/P" +
                        std::to_string(clsNum) + "/";
  if(!fs::is_directory(folder)) {
    fs::path p(folder);
    fs::create_directory(p);
  }
  cv::imwrite(folder + "im100" + std::to_string(imNum) + ".png", im100);


  folder = "Gestures/dynamic_poses/uncut/norm_depth/P" +
                        std::to_string(clsNum) + "/";
  if(!fs::is_directory(folder)) {
    fs::path p(folder);
    fs::create_directory(p);
  }
  cv::imwrite(folder + "imNDp" + std::to_string(imNum) + ".png", imNormTotal);

  return true;
}


int
main(int argc, char **argv)
{
  auto device = rscg::CameraDeviceRSWindows();
  rscg::GraphProbs graph{200};

  bool setToStop = false;

  cv::Mat frame                   {480, 640, CV_8UC1  ,  cv::Scalar(0)},
          frameColor              {480, 640, CV_8UC3  ,  cv::Scalar(0)},
          frame16C                {480, 640, CV_16UC1 ,  cv::Scalar(0)},
          frameNormDepthRng       {480, 640, CV_8UC1  ,  cv::Scalar(0)},
          frameNormDepthRngColor  {480, 640, CV_8UC3  ,  cv::Scalar(0)},
          graphIm                 {480, 640, CV_8UC3  ,  cv::Scalar(0)},
          im100x100               {100, 100, CV_8UC1  ,  cv::Scalar(0)},
          im100x100Color          {100, 100, CV_8UC3  ,  cv::Scalar(0)};

  int thresh = 200;

  cv::namedWindow("frame", 0);
  cv::createTrackbar("thresh", "frame", &thresh, 1000);

  std::vector<int> imToClassify;
  imToClassify.resize(50 * 50 * 3);

  uint16_t value = 0, depthCloserCam = 9999;
  const uint16_t maxDepth = 32727;
  uint8_t normValue = 0;

  sf::TcpSocket sock;
  bool connected = true;
  sf::Socket::Status status = sock.connect("127.0.0.1", 31000);
  if(status != sf::Socket::Done) {
    std::cerr << "erro on connecting" << std::endl;
    connected = false;
  }

  std::vector<float> resClass;

  size_t cont = 0;

  bool recording = false;

  cv::Mat fullIm{480 * 2, 640 * 3, CV_8UC3, cv::Scalar(0)};

  int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 1;
  int thickness = 3;
  int baseline;

  int clsNum = 1;

  bool isSavingVideos = false;

  while(!setToStop || recording) {
    device.fetchDepthFrame();

    // find the closest depth from camera
    depthCloserCam = 9999;
    for(const auto &it : device.getDepthFrame1Chanels()) {
      if(it != 0 && it < depthCloserCam) {
        depthCloserCam = it;
      }
    }

    for(size_t y = 0; y < 640; ++y) {
      for(size_t x = 0; x < 480; ++x) {
        value = device.getDepthFrame1Chanels()[x * 640 + y];

        frame.at<uint8_t>(x, y) = value % 255;
        frame16C.at<uint16_t>(x, y) = 
          ((value < (uint16_t)thresh + depthCloserCam) && (value > 10) ?
            value % 255 : 0);

        // normalizing using linear interpolation
        // normValue = (uint8_t)((255.00) * ((double)value / (double)maxDepth));

        frameNormDepthRng.at<uint8_t>(x, y) =
          ( (value < (uint16_t)thresh + depthCloserCam) ? 
              value % 255: 0);
      }
    }
    
    char reskey = (char)cv::waitKey(60);
    if(reskey == 'q') {
      setToStop = true;
    }
    else if(reskey == 'c') {
      ++clsNum;
      cont = amountImgUncutClsFolder(clsNum, "norm_depth");
    }
    else if(reskey == 'v') {
      --clsNum;
      cont = amountImgUncutClsFolder(clsNum, "norm_depth");
    }
    else if(reskey == ' ') {
      recording = !recording;
    }

    cv::resize(frameNormDepthRng, im100x100, cv::Size{100, 100}, 0, 0, 
               CV_INTER_AREA);

    cv::cvtColor(frame, frameColor, cv::COLOR_GRAY2BGR);
    frameColor.copyTo(fullIm(cv::Rect(0, 0, frameColor.cols, 
                                      frameColor.rows)));
    cv::cvtColor(frameNormDepthRng, frameNormDepthRngColor, 
                 cv::COLOR_GRAY2BGR);
    frameNormDepthRngColor.copyTo(fullIm(cv::Rect(640, 0, 
                                         frameNormDepthRngColor.cols,
                                         frameColor.rows)));

    cv::cvtColor(im100x100, im100x100Color, cv::COLOR_GRAY2BGR);
    im100x100Color.copyTo(fullIm(cv::Rect(2*640, 0, im100x100Color.cols, 
                                       im100x100Color.rows)));
                                       
    graphIm.copyTo(fullIm(cv::Rect(0, 480, graphIm.cols,
                                   graphIm.rows)));

    cv::Size textSize = cv::getTextSize(std::to_string(cont), fontFace,
                                        fontScale, thickness, &baseline);
    cv::putText(fullIm, std::to_string(cont), {0, 25}, fontFace, fontScale,
                cv::Scalar(0, 128, 0), 1, 1);

    textSize = cv::getTextSize(std::string("cls name:")
                               + std::to_string(clsNum), fontFace, fontScale,
                               thickness, &baseline);
    cv::putText(fullIm, std::string("cls name:") + std::to_string(clsNum),
                {0, 75}, fontFace, fontScale, cv::Scalar(0, 128, 0), 1, 1);

    textSize = cv::getTextSize(std::string("curr seq:")
                               + std::to_string(cont), fontFace, fontScale,
                               thickness, &baseline);
    cv::putText(fullIm, std::string("curr seq:") + std::to_string(cont),
                {0, 125}, fontFace, fontScale, cv::Scalar(0, 128, 0), 1, 1);

    cv::imshow("frame", fullIm);
    
    if(recording) {
      std::async(std::launch::async, saveSingleImage, 
                 frame16C, im100x100, frameNormDepthRng, cont, clsNum);
      cont++;
    }
  }

  if(connected)
    sock.disconnect();

  return 0;
}
