#include <iostream>
#include <fstream>
#include <memory>
#include <climits>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <cstdio>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <SFML/Network.hpp>

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
               cv::Size imSize)
{
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

  for(size_t i = 0; i < 14; ++i)
  {
    float a = std::stof(str.substr(i * 9, 7));
    resultVec.push_back(a);
  }
  
  return resultVec;
} 


class ImageClassifier
{
public:
  ImageClassifier();

  const std::vector<float>& classifyImage(const cv::Mat &im);
private:
  sf::Socket _socket;
};

cv::Mat cutImage(const cv::Mat& im, const cv::Size &size);

void 
fillTexture(SDL_Texture * texture, cv::Mat const &mat)
{
  IplImage * img = &(IplImage)mat;

  unsigned char * texture_data = NULL;
  int texture_pitch = 0;

  SDL_LockTexture(texture, 0, (void **)&texture_data, &texture_pitch);
  memcpy(texture_data, (void *)img->imageData,
         img->width * img->height * img->nChannels);
  SDL_UnlockTexture(texture);
}

int
main(int argc, char **argv)
{
  uint16_t currMax = maxDepth, normValue = 0;
  auto device = rscg::CameraDeviceWindows();
  rscg::GraphProbs graph{200};

  bool setToStop = false;

  cv::Mat frame          {480, 640, CV_8UC1,  cv::Scalar(0)},
          frameColor     {480, 640, CV_8UC3,  cv::Scalar(0)},
          dist           {480, 640, CV_16UC1, cv::Scalar(0)},
          distColor      {480, 640, CV_8UC3,  cv::Scalar(0)},
          frame2WithRect {480, 640, CV_8UC3,  cv::Scalar(0)},
          color          {480, 640, CV_8UC3,  cv::Scalar(0)}, 
          frame2         {480, 640, CV_8UC1,  cv::Scalar(0)},
          frame2Color    {480, 640, CV_8UC3,  cv::Scalar(0)},
          graphIm        {480, 640, CV_8UC3,  cv::Scalar(0)},
          fDepth         {480, 640, CV_8UC1,  cv::Scalar(0)},
          im5050         { 50,  50, CV_8UC1,  cv::Scalar(0)},
          im5050Color    { 50,  50, CV_8UC3,  cv::Scalar(0)},
          fDepthColor    {480, 640, CV_8UC3,  cv::Scalar(0)};

  int thresh = 200;

  double min, max;
  cv::Point min_loc, max_loc;

  cv::namedWindow("frame", 0);
  cv::createTrackbar("thresh", "frame", &thresh, 1000);

  std::vector<uint16_t> imDepth;
  std::vector<uint8_t> imColor;

  std::vector<int> imToClassify;
  imToClassify.resize(50 * 50 * 3);

  std::vector<cv::Mat> imSeq;

  imSeq.resize(3000);

  uint16_t value = 0;

  sf::TcpSocket sock;
  bool connected = true;
  sf::Socket::Status status = sock.connect("127.0.0.1", 31000);
  if(status != sf::Socket::Done)
  {
    std::cerr << "erro on connecting" << std::endl;
    connected = false;
  }

  std::vector<float> resClass;

  int cont = 0;
  rscg::Rect<int> boundingBox;

  bool recording = false;

  int contFiles = 0;

  cv::Mat h;
  cv::Mat fullIm{480*2, 640*3, CV_8UC3, cv::Scalar(0)};

  sf::Clock clk;
  clk.restart();

  int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 1;
  int thickness = 3;
  int baseline;

  std::future<bool> saveSeqFut;
  int currSeqNum = 1;
  int folderNum = 1;
  int clsNum = 1;
  uint32_t imsInSeq = 0;
  std::string testOrTrain = "train";

  while(! setToStop || recording)
  {
    device.fetchDepthFrame();
    device.fetchColorFrame();

    imDepth = device.getDepthFrame1Chanels();

    currMax = maxDepth;
    
    uint16_t currMinDepth = 9999;
    for(const auto &it : imDepth)
    {
      if(it != 0 && it < currMinDepth)
        currMinDepth = it;
    }

    for(size_t y = 0; y < 640; ++y)
    {
      for(size_t x = 0; x < 480; ++x)
      {
        value = imDepth[x * 640 + y];
        if(maxDepth < value)
          maxDepth = value;

        normValue = (uint8_t)((255.00) * ((double)value / (double)maxDepth));

        frame.at<uint8_t>(x, y) = 
          (uint8_t)(( (value < (uint16_t)thresh + currMinDepth) 
                    && (value > 10)) ? 255 : 0);

        frame2.at<uint8_t>(x, y) = (((value < (uint16_t)thresh + currMinDepth)
                                     && (value > 10)) ? value : 0);
        fDepth.at<uint8_t>(x, y) = normValue;

      }
    }

    char reskey = (char)cv::waitKey(60);
    if(reskey == 'q')
      setToStop = true;
    if(!recording && reskey == 't')
      testOrTrain = (testOrTrain == "train" ? "test" : "train");
    if(reskey == ' ')
    {
      recording = !recording;
      if(!recording)
      {
        saveSeqFut = std::async(std::launch::async, 
                                [&folderNum, &testOrTrain, 
                                 &clsNum]
                                (const std::vector<cv::Mat>& imgs,
                                 uint32_t size, int seqNum) -> bool
        {
          std::string folderName = "Gestures/dynamic_poses/F" +
                                    std::to_string(folderNum) + "/"
                                    + testOrTrain +
                                    "/P" + std::to_string(clsNum) +
                                    "/e" + std::to_string(seqNum) + "/";
          std::cout << "saving to folder" << folderName << '\n';
          for(int i = 0; i < size; ++i)
            cv::imwrite(folderName + "im" + std::to_string(i) + ".jpg",
                        imgs[i]);
          std::cout << "finished printing \n";
          return true;
        }, imSeq, cont, currSeqNum);

        cont = 0;
        ++currSeqNum;
      }
    }

    frame2WithRect = frame2.clone();
    boundingBox = rscg::boundingSquare(frame2WithRect);
    cv::rectangle(frame2WithRect, {boundingBox.x, boundingBox.y, boundingBox.w,
                                   boundingBox.h}, cv::Scalar(128));
    if(boundingBox.w > 0)
    {
      cv::Mat boundedFrame2 = cv::Mat::zeros(cv::Size{boundingBox.w,
                                                      boundingBox.h}, CV_8UC1);
      h = boundedFrame2;
      size_t y2, y1, x2, x1;
      y2 = 0;
      y1 = boundingBox.y;
      for(; y2 < boundingBox.h; ++y2, ++y1)
      {
        x2 = 0;
        x1 = boundingBox.x;
        for(; x2 < boundingBox.w; ++x2, ++x1)
        {
          float v = 0;
          v = frame2.at<uint8_t>(std::min((size_t)479, y1),
                                 std::min((size_t)639, x1));

          boundedFrame2.at<uint8_t>(y2, x2) = v;
        }
      }

      cv::resize(boundedFrame2, im5050, {50, 50}, 0, 0, CV_INTER_AREA);

      for(size_t y = 0; y < 50; ++y)
      {
        for(size_t x = 0; x < 50; ++x)
          imToClassify[x + y * 50] = im5050.at<uint8_t>(y, x);
      }
    }

    auto cir = frame;

    if(recording)
      cv::circle(cir, max_loc, 10, cv::Scalar(128));
    if(boundingBox.h != -1 && connected)
    {
      resClass = classifyImgNet(sock, imToClassify, {50, 50});
      graph.update(resClass);
      graphIm = graph.render();
    }

    cv::cvtColor(frame, frameColor, cv::COLOR_GRAY2BGR);
    frameColor.copyTo(fullIm(cv::Rect(640, 0, frameColor.cols, 
                      frameColor.rows)));

    cv::cvtColor(fDepth, fDepthColor, cv::COLOR_GRAY2BGR);
    fDepthColor.copyTo(fullIm(cv::Rect(0, 0, frameColor.cols,
                               frameColor.rows)));
    
    cv::cvtColor(im5050, im5050Color, cv::COLOR_GRAY2BGR);
    im5050Color.copyTo(fullIm(cv::Rect(2*640, 0, im5050Color.cols, 
                                       im5050Color.rows)));
                                       
    graphIm.copyTo(fullIm(cv::Rect(0, 480, graphIm.cols,
                                   graphIm.rows)));
                                   
    cv::Size textSize = cv::getTextSize(std::to_string(cont), fontFace,
                                        fontScale, thickness, &baseline);
    cv::putText(fullIm, std::to_string(cont), {0, 20}, fontFace, fontScale,
                cv::Scalar(0, 128, 0), 1, 1);

    cv::imshow("frame", fullIm);
    
    if(recording && boundingBox.w > 0)
    {
      imSeq[cont] = im5050.clone();
      cont++;
    }
  }

  if(connected)
    sock.disconnect();

  return 0;
}
