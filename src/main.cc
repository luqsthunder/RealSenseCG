#include <iostream>
#include <fstream>
#include <memory>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <chrono>

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

template<typename T>
inline void
writePPM(const std::string &name, const cv::Mat &im, const cv::Size &size,
         T maxval) {
  std::ofstream file;
  file.open(name, std::ios::app);
  unsigned ch = im.channels();
  file << "P" + std::to_string(ch) << '\n';
  file << T(size.width) << ' ' << T(size.height) << '\n';
  file << T(maxval) << '\n';

  T value = T();

  for(size_t y = 0; y < im.cols; ++y) {
    for(size_t x = 0; x < im.rows; ++x) {
      value = im.at<T>(x, y);
      for(uint32_t cIt = 0; cIt < ch; ++cIt) {
        file << value << ' ';
      }
      if(ch > 1) {
        file << ' ';
      }
    }
    if(y < size.height) file << '\n';
  }
  file.close();
}

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

inline void 
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

inline void 
saveVideos(const int currentTalk, const int sample, int totalFrames,
           const std::vector<cv::Mat> &frames, int videoFps,
           const std::vector<std::vector<Joint>> &jointsFrames) {
  namespace fs = boost::filesystem;
  std::string folderPPM = "Gestures/Videos/uncut/talk" + std::to_string(currentTalk);

  if(!fs::is_directory(folderPPM)) {
    fs::path p(folderPPM);
    fs::create_directory(p);
  }

  folderPPM += "/sample" + std::to_string(sample);
  if(!fs::is_directory(folderPPM)) {
    fs::path p(folderPPM);
    fs::create_directory(p);
  }

  std::string mainFolder = folderPPM;
  folderPPM += "/frames";
  if(!fs::is_directory(folderPPM)) {
    fs::path p(folderPPM);
    fs::create_directory(p);
  }

  char filename[250];
  for(size_t i = 0; i < totalFrames; ++i) {
    sprintf(filename, "%s/frame_%04d.ppm",folderPPM.c_str(), i);
    cv::imwrite(filename, frames[i]);
  }

  std::vector<std::string> csvHeaders = {
    "JointType_SpineBase", "JointType_SpineMid", "JointType_Neck", 
    "JointType_Head", "JointType_ShoulderLeft", "JointType_ElbowLeft",
    "JointType_WristLeft", "JointType_HandLeft", "JointType_ShoulderRight",
    "JointType_ElbowRight", "JointType_WristRight", "JointType_HandRight",
    "JointType_HipLeft", "JointType_KneeLeft", "JointType_AnkleLeft",
    "JointType_FootLeft", "JointType_HipRight", "JointType_KneeRight",
    "JointType_AnkleRight", "JointType_FootRight", "JointType_SpineShoulder",
    "JointType_HandTipLeft", "JointType_ThumbLeft", "JointType_HandTipRight",
    "JointType_ThumbRight",
  };
  std::ofstream csvFile;
  csvFile.open(mainFolder + "/skelleton_" + std::to_string(sample) + ".csv", 
               std::ios::app);
  for(size_t i = 0; i < csvHeaders.size(); ++i) {
    csvFile <<  csvHeaders[i] <<  (i < csvHeaders.size() - 1) ? ", " : "";
  }
  csvFile << '\n';
  
  for(size_t i = 0; i < jointsFrames.size(); ++i) {
    for(size_t i2 = 0; i2 < jointsFrames[i].size(); ++i2) {
      csvFile << "\"" 
              << jointsFrames[i][i2].Position.X << ", "
              << jointsFrames[i][i2].Position.Y << ", " 
              << jointsFrames[i][i2].Position.Z << 
              ((i2 < 24) ? "\", " : "\"");
    }
    csvFile << '\n';
  }
  csvFile.close();
}

int
main(int argc, char **argv) {
  auto device = rscg::CameraDeviceKinect();
  bool setToStop = false;

  cv::namedWindow("frame", 0);
  cv::namedWindow("color", 0);

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

  std::future<void> waitSave;

  std::vector<float> resClass;

  bool recording = false;

  int fontFace = cv::HersheyFonts::FONT_HERSHEY_PLAIN;
  double fontScale = 1;
  int thickness = 3;
  int baseline;

  int clsNum = 1, currTalk = 1, sample = 1;

  bool saving = false;

  int width = 512, height = 424;

  cv::Mat allScreen{height + 128, width, CV_8UC3, cv::Scalar(0, 0, 0)};
  cv::Mat recColor{128, width / 2, CV_8UC3, cv::Scalar(0, 0, 0)};
  cv::Mat trackingSkellColor{128, width / 2, CV_8UC3, cv::Scalar(0, 0, 0)};

  std::vector<cv::Mat> videoDepth;
  std::vector<std::vector<Joint>> joints;

  uint32_t maxSeg = 60;

  joints.reserve(30 * maxSeg);
  for(int i = 0; i < 30 * maxSeg; ++i) {
    videoDepth.push_back(cv::Mat{width, height, CV_16UC1, cv::Scalar(0)});
  }

  unsigned cont = 0, samplesCount = 0;

  unsigned fpsCount = 0, currFps = 0;
  std::chrono::time_point<std::chrono::steady_clock> lastTime, currTime;
  lastTime = currTime = std::chrono::steady_clock::now();

  std::vector<JointType> neededJoints{
    JointType_Head, JointType_Neck, JointType_HandLeft, JointType_HandRight,
    JointType_SpineShoulder, JointType_ShoulderLeft, JointType_ShoulderRight,
    JointType_ElbowLeft, JointType_ElbowRight, JointType_SpineMid, 
    JointType_SpineBase, JointType_WristLeft, JointType_WristRight, 
    JointType_ThumbLeft/*, JointType_ThumbRight, JointType_HandTipLeft,
    JointType_HandTipRight*/
  };

  bool jointsNeededTracked = false;

  while(!setToStop || recording || saving) {
    device.fetchDepthFrame();
    device.fetchColorFrame();
    device.fetchSkeleton();
    
    jointsNeededTracked = device.isThatJointsTracked(neededJoints);
    if(jointsNeededTracked) {
      trackingSkellColor = cv::Scalar(0, 255, 0);
    }
    else {
      trackingSkellColor = cv::Scalar(255);
    }

    char reskey = (char)cv::waitKey(1);
    if(reskey == 'q') { 
      setToStop = true; 
    }
    else if(reskey == 't') {
      ++currTalk;
    }
    else if(reskey == 'r') {
      --currTalk;
    }
    else if(reskey == ' ') {
      if(!saving && jointsNeededTracked) {
        recording = !recording;

        if(recording) {
          recColor = cv::Scalar(0, 0, 255);
        }
        else {
          saving = true;
          waitSave = std::async(std::launch::async, saveVideos,

                                currTalk, sample, cont, videoDepth, 15, 
                                joints);
        }
      }
    }
   
    if(recording) {
      videoDepth[cont] = device.getDepthFrame1Chanels().clone();
      joints.push_back(device.getSkeletonJointVec());
      ++cont;
      if(cont >= 30 * maxSeg) {
        recording = false;
        saving = true;
      }
    }
    
    if(waitSave.valid()) {
      if(waitSave.wait_for(std::chrono::seconds(0)) !=
         std::future_status::ready) {
        recColor = cv::Scalar(0, 255, 0);
        recording = false;
        saving = true;
      }
      else if(waitSave.wait_for(std::chrono::seconds(0)) ==
              std::future_status::ready) {
        waitSave.get();
        saving = false;
        recording = false;
        /*joints.clear();
        joints.reserve(30 * maxSeg);
        videoDepth.clear();
        for(int i = 0; i < 30 * maxSeg; ++i) {
          videoDepth.push_back(cv::Mat{width, height, CV_16UC1, 
                                       cv::Scalar(0)});
        }*/
        cont = 0;
        sample += 1;
      }
    }

    if(recording) {
      if(!jointsNeededTracked && !saving) {
        recording = false;
        cont = 0;
      }
    }
    else if(!saving && !recording) {
      recColor = cv::Scalar(0, 255, 255);
    }

    device.renderSkeletonJointsToDepth();
    
    device.getDepthFrame3Chanels()
          .copyTo(allScreen(cv::Rect(0, 0, device.getDepthFrame3Chanels().cols, 
                                     device.getDepthFrame3Chanels().rows)));

    recColor.copyTo(allScreen(cv::Rect(0, height, recColor.cols,
                                       recColor.rows)));

    trackingSkellColor.copyTo(allScreen(cv::Rect(width / 2, height,
                                                 trackingSkellColor.cols, 
                                                 trackingSkellColor.rows)));
                                                 
    std::string currFPSStr = "FPS: " + std::to_string(currFps);
    cv::Size textSize = cv::getTextSize(currFPSStr,  fontFace, fontScale, 
                                        thickness, &baseline);
    cv::putText(allScreen, currFPSStr, {0, 25}, fontFace, fontScale,
                cv::Scalar(0, 255, 255), 1, 1);

    cv::imshow("frame", allScreen);
    cv::imshow("color", device.getColorFrame());
    ++fpsCount;
    currTime = std::chrono::steady_clock::now();
    using seconds = std::chrono::milliseconds;
    auto elps = std::chrono::duration_cast<seconds>(currTime - lastTime);
    if(elps.count() >= 1000) {
      currFps = fpsCount;
      fpsCount = 0;
      lastTime = currTime;
    }

  }

  cv::destroyAllWindows();

  if(connected)
    sock.disconnect();

  return 0;
}
