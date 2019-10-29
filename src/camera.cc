#include "camera.h"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <climits>
#include <strsafe.h>


using namespace rscg;

CameraDeviceKinect::CameraDeviceKinect()
  : _paDepthReaderFrame(nullptr),
    _skellTracked(false),
    _paSensor(nullptr),
    _paSkellFrameReader(nullptr),
    _paCoordinateMapper(nullptr),
    _paColorFrameReader(nullptr),
    _imdepth{424, 512, CV_16UC1, cv::Scalar(0)}, 
    _imdepth3{424, 512, CV_8UC3, cv::Scalar(0,0,0)},
    _imdepth1caux{424, 512, CV_8UC1, cv::Scalar(0)},
    _imColor{1080, 1920, CV_8UC4, cv::Scalar(0, 0, 0)},
    _joints(JointType_Count),
  _bones({
    //torso
    {JointType_Head, JointType_Neck},
    {JointType_Neck, JointType_SpineShoulder},
    {JointType_SpineShoulder, JointType_SpineMid},
    {JointType_SpineMid, JointType_SpineBase},
    {JointType_SpineShoulder, JointType_ShoulderRight},
    {JointType_SpineShoulder, JointType_ShoulderLeft},
    {JointType_SpineBase, JointType_HipRight},
    {JointType_SpineBase, JointType_HipLeft},

    // Right Arm    
    {JointType_ShoulderRight, JointType_ElbowRight},
    {JointType_ElbowRight, JointType_WristRight},
    {JointType_WristRight, JointType_HandRight},
    {JointType_HandRight, JointType_HandTipRight},
    {JointType_WristRight, JointType_ThumbRight},

    // Left Arm
    {JointType_ShoulderLeft, JointType_ElbowLeft},
    {JointType_ElbowLeft, JointType_WristLeft},
    {JointType_WristLeft, JointType_HandLeft},
    {JointType_HandLeft, JointType_HandTipLeft},
    {JointType_WristLeft, JointType_ThumbLeft},

    // Right Leg
    {JointType_HipRight, JointType_KneeRight},
    {JointType_KneeRight, JointType_AnkleRight},
    {JointType_AnkleRight, JointType_FootRight},

    // Left Leg
    {JointType_HipLeft, JointType_KneeLeft},
    {JointType_KneeLeft, JointType_AnkleLeft},
    {JointType_AnkleLeft, JointType_FootLeft}}) {
  HRESULT hr = GetDefaultKinectSensor(&_paSensor);
  if(FAILED(hr)) {
    std::cerr << "error on initializing kinect \n";
    return;
  }

  // Initialize the Kinect and get coordinate mapper and the body reader
  IBodyFrameSource* paBodyFrameSrc = nullptr;
  IDepthFrameSource *paDepthFrameSrc = nullptr;
  IColorFrameSource *paColorFrameSrc = nullptr;

  hr = _paSensor->Open();
  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_DepthFrameSource(&paDepthFrameSrc);
    if(SUCCEEDED(hr)) {
      hr = paDepthFrameSrc->OpenReader(&_paDepthReaderFrame);
    }
    else {
      std::cerr << "cannot create depth frame reader\n";
      return;
    }
  }
  else {
    std::cerr << "cannot create depth frame source\n";
    return;
  }

  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_ColorFrameSource(&paColorFrameSrc);
    if(SUCCEEDED(hr)) {
      hr = paColorFrameSrc->OpenReader(&_paColorFrameReader);
    }
    else {
      std::cerr << "cannot create color frame reader\n";
      return;
    }
  }
  else {
    std::cerr << "cannot create color frame source\n";
    return;
  }

  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_BodyFrameSource(&paBodyFrameSrc);
    if(SUCCEEDED(hr)) {
      hr = paBodyFrameSrc->OpenReader(&_paSkellFrameReader);
    }
    else {
      std::cerr << "cannot create skelleton frame reader\n";
      return;
    }
  }
  else {
    std::cerr << "cannot create skelleton frame source\n";
    return;
  }

  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_CoordinateMapper(&_paCoordinateMapper);
  }
  else {
    std::cerr << "cannot create coordinate mapping\n";
    return;
  }

  if(paDepthFrameSrc != nullptr) {
    paDepthFrameSrc->Release();
  }
  else {
    std::cerr << "cannot release Depth source\n";
    return;
  }

  if(paColorFrameSrc!= nullptr) {
    paColorFrameSrc->Release();
  }
  else {
    std::cerr << "cannot release Color source\n";
    return;
  }

  if(paBodyFrameSrc != nullptr) {
    paBodyFrameSrc->Release();
  }
  else {
    std::cerr << "cannot release Skeleton source\n";
    return;
  }

  if(FAILED(hr) || _paSensor == nullptr) {
    std::cerr << "no ready kinect found \n";
    return;
  }
}

const cv::Point2f
CameraDeviceKinect::worldToScreenPoint(const CameraSpacePoint& bodyPoint,
                                       const cv::Size winSz) {
  // Calculate the body's position on the screen
  DepthSpacePoint depthPoint = {0};
  _paCoordinateMapper->MapCameraPointToDepthSpace(bodyPoint, &depthPoint);

  float screenPointX = static_cast<float>(depthPoint.X * winSz.width) / 424;
  float screenPointY = static_cast<float>(depthPoint.Y * winSz.height) / 512;

  return {screenPointX, screenPointY};
}

CameraDeviceKinect::~CameraDeviceKinect() {
  if(_paDepthReaderFrame != nullptr) {
    _paDepthReaderFrame->Release();
  }

  if(_paSkellFrameReader != nullptr) {
    _paSkellFrameReader->Release();
  }

  if(_paCoordinateMapper!= nullptr) {
    _paCoordinateMapper->Release();
  }

  if(_paColorFrameReader != nullptr) {
    _paColorFrameReader->Release();
  }

  if(_paSensor != nullptr) {
    _paSensor->Close();
    _paSensor->Release();
  }
}

bool
CameraDeviceKinect::allJointsTracked() {
  return _allJointsTracked;
}

void
CameraDeviceKinect::fetchDepthFrame()
{
  if(_paDepthReaderFrame == nullptr) {
    return;
  }

  IDepthFrame *paFrame;
  HRESULT hr = _paDepthReaderFrame->AcquireLatestFrame(&paFrame);
  bool notGet = FAILED(hr);
  while(notGet) {
    hr = _paDepthReaderFrame->AcquireLatestFrame(&paFrame);
    notGet = FAILED(hr);
  }

  if(SUCCEEDED(hr)) {
    int nWidth, nHeigth;
    IFrameDescription* paFrameDescription = NULL;

    hr = paFrame->get_FrameDescription(&paFrameDescription);

    if(SUCCEEDED(hr)) {
      hr = paFrameDescription->get_Width(&nWidth);
    }

    if(SUCCEEDED(hr)) {
      hr = paFrameDescription->get_Height(&nHeigth);
    }

    if(SUCCEEDED(hr)) {
      hr = paFrame->CopyFrameDataToArray(nWidth * nHeigth,
                                         (UINT16*)(_imdepth.data));
      cv::cvtColor( ((_imdepth - cv::Scalar(200)) * 255 / 4800), _imdepth3,
                   cv::COLOR_GRAY2BGR);
    }

    paFrameDescription->Release();
  }
  else {
    std::cout << "failed \n";
  }

  if(paFrame != nullptr) {
    paFrame->Release();
  }
}

void 
CameraDeviceKinect::fetchColorFrame() {
  IColorFrame *paColorFrame = nullptr;
  HRESULT hr = _paColorFrameReader->AcquireLatestFrame(&paColorFrame);
  ColorImageFormat imageFormat = ColorImageFormat_None;
  UINT nBufferSize = 0;
  RGBQUAD *pBuffer = NULL;
  int nWidth = 0;
  int nHeight = 0;

  if(SUCCEEDED(hr)) {
    IFrameDescription *paFrameDescription = nullptr;
    hr = paColorFrame->get_FrameDescription(&paFrameDescription);
    if(SUCCEEDED(hr))
    {
      hr = paFrameDescription->get_Width(&nWidth);
    }

    if(SUCCEEDED(hr))
    {
      hr = paFrameDescription->get_Height(&nHeight);
    }

    if(SUCCEEDED(hr))
    {
      hr = paColorFrame->get_RawColorImageFormat(&imageFormat);
    }

    if(SUCCEEDED(hr))
    {
      if(imageFormat == ColorImageFormat_Bgra) {
        hr = paColorFrame->AccessRawUnderlyingBuffer(&nBufferSize,
                                                     reinterpret_cast<BYTE**>(&pBuffer));
      }
      else {
        hr = paColorFrame->CopyConvertedFrameDataToArray(nWidth * nHeight *
                                                         sizeof(RGBQUAD),
                                                         reinterpret_cast<BYTE*>(_imColor.data),
                                                         ColorImageFormat_Bgra);
      }

      if(SUCCEEDED(hr)) {
      }
    }
    if(paFrameDescription != nullptr) {
      paFrameDescription->Release();
    }
  }

  if(paColorFrame != nullptr) {
    paColorFrame->Release();
  }
}

void
CameraDeviceKinect::fetchSkeleton() {
  IBodyFrame* pBodyFrame = NULL;

  _skellTracked = false;
  HRESULT hr = _paSkellFrameReader->AcquireLatestFrame(&pBodyFrame);
  bool notGet = FAILED(hr);
  while(notGet) {
    hr = _paSkellFrameReader->AcquireLatestFrame(&pBodyFrame);
    notGet = FAILED(hr);
  }

  if(SUCCEEDED(hr)) {
    _currentTime = 0;
    hr = pBodyFrame->get_RelativeTime(&_currentTime);

    IBody* ppBodies[BODY_COUNT] = {nullptr};
    if(SUCCEEDED(hr)) {
      hr = pBodyFrame->GetAndRefreshBodyData(_countof(ppBodies), ppBodies);
    }

    if(SUCCEEDED(hr)) {
      for(int i = 0; i < BODY_COUNT; ++i) {
        IBody* pBody = ppBodies[i];
        if(pBody != nullptr) {
          BOOLEAN bTracked = false;
          hr = pBody->get_IsTracked(&bTracked);

          if(SUCCEEDED(hr) && bTracked) { 
            hr = pBody->GetJoints((UINT)_joints.size(), _joints.data());
            _skellTracked = true;
          }
        }
      }
    }

    _allJointsTracked = true;
    for(const auto &i : _joints) {
      if(i.TrackingState != TrackingState_Tracked) {
        _allJointsTracked = false;
        break;
      }
    }

    for(int i = 0; i < _countof(ppBodies); ++i) {
      if(ppBodies[i] != nullptr) {
        ppBodies[i]->Release();
      }
    }
  }
  if(pBodyFrame != nullptr) {
    pBodyFrame->Release();
  }
}

bool 
CameraDeviceKinect::isThatJointsTracked(const std::vector<JointType> &j) {
  bool track = true;
  for(const auto &it : j) {
    if(_joints[it].TrackingState == TrackingState_NotTracked) {
      track = false;
      break;
    }
  }

  return track;
}

void
CameraDeviceKinect::renderSkeletonJointsToDepth() {
  if(!_skellTracked) {
    return;
  }

  cv::Point_<unsigned> screenCoord1, screenCoord2;

  for(const auto &b : _bones) {
    screenCoord1 = worldToScreenPoint(_joints[b.first].Position, {424, 512});
    screenCoord2 = worldToScreenPoint(_joints[b.second].Position, {424, 512});

    bool jTrack = (_joints[b.first].TrackingState == TrackingState_Tracked &&
                   _joints[b.second].TrackingState == TrackingState_Tracked);
    cv::Scalar colorBone = jTrack ? cv::Scalar{0, 255, 255} :
                                    cv::Scalar{0, 0, 0};
    int thick = jTrack ? 2 : 1;

    cv::line(_imdepth3, screenCoord1, screenCoord2, colorBone, thick);
  }
}

const cv::Mat& 
CameraDeviceKinect::getColorFrame() {
  return _imColor;
}

const int64_t
CameraDeviceKinect::getCurrentTimeSkeletonFrame() {
  return _currentTime;
}

const std::vector<Joint>&
CameraDeviceKinect::getSkeletonJointVec() {
  return _joints;
}

const cv::Mat&
CameraDeviceKinect::getDepthFrame3Chanels() {
  return _imdepth3;
}

const cv::Mat&
CameraDeviceKinect::getDepthFrame1Chanels() {
  return _imdepth;
}
