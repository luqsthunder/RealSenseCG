#include "camera.h"

#include <opencv2/imgproc.hpp>
#include <iostream>
#include <climits>
#include <strsafe.h>


using namespace rscg;

CameraDeviceRSWindows::CameraDeviceRSWindows() : 
  sm(nullptr),
  _imgdepth1c{480, 640, CV_16UC1, cv::Scalar(0)},
  _imgdepth3{480, 640, CV_8UC3, cv::Scalar(0,0,0)} {
  pxcStatus status;

  msize = {640, 480};
  sm = PXCSenseManager::CreateInstance();

  if(!sm) {
    throw std::runtime_error("could not create camera manager");
  }

  sm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, 640, 480, 60);
  sm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, 640, 480, 60);
  status = sm->Init();

  auto device = sm->QueryCaptureManager()->QueryDevice();

  auto mirrorMode = PXCCapture::Device::MirrorMode::MIRROR_MODE_HORIZONTAL;
  device->SetMirrorMode(mirrorMode);
  auto focus = device->QueryDepthFocalLength();
  auto pp = device->QueryDepthPrincipalPoint();

  _intri = Intrinsics{pp.x, pp.y, focus.x, focus.y};
}

CameraDeviceRSWindows::~CameraDeviceRSWindows() {
  sm->Release();
}

void 
CameraDeviceRSWindows::fetchDepthFrame() {
  /// This function blocks until a sample is ready
  if(sm->AcquireFrame(true) < PXC_STATUS_NO_ERROR)
    std::cerr << "could not aquire frame" << std::endl;

  auto streamtype = PXCCapture::StreamType::STREAM_TYPE_DEPTH;

  PXCCapture::Sample *sample = sm->QuerySample();
  auto image = sample->depth;

  PXCImage::ImageData imageData;
  PXCImage::ImageInfo imageInfo = image->QueryInfo();

  /// Below is a common AcquireAccess - Process - ReleaseAccess pattern
  /// for RealSense images.
  pxcStatus status;

  pxcI32 sizeX = imageInfo.width;  // Number of pixels in X
  pxcI32 sizeY = imageInfo.height;  // Number of pixels in Y

  status = image->AcquireAccess(PXCImage::ACCESS_READ, &imageData);

  if(imageData.pitches[0] != 2 * sizeX) {
    throw std::runtime_error("Unexpected data in buffer");
  }

  const pxcI32 pixelPitch = 2;

  if(status == PXC_STATUS_NO_ERROR) {
    /// becouse every pixel is store as 2 bytes, and obviously pitch is 2
    /// times width, changing array pointer to a double size will work 
    /// equals as manipulating bytes

    //pxcU16* imageArray = (pxcU16*)imageData.planes[0];
    uint16_t* imageArray = (uint16_t*)imageData.planes[0];
    //pxcBYTE* imageArray = imageData.planes[0];

    for(pxcI32 y = 0; y < sizeY; y++) {
      for(pxcI32 x = 0; x < sizeX; x++) {
        uint16_t depth = imageArray[x + y * sizeX];


        _imgdepth3.at<cv::Vec3b>({x, y}) = {(uint8_t)(depth % 256), 
                                            (uint8_t)(depth % 256),
                                            (uint8_t)(depth % 256)};

        _imgdepth1c.at<uint16_t>(cv::Point{x, y}) = {depth};
      }
    }
  }
  image->ReleaseAccess(&imageData);
  sm->ReleaseFrame();
}

const cv::Mat&
CameraDeviceRSWindows::getDepthFrame3Chanels() {
  return _imgdepth3;
}

const cv::Mat&
CameraDeviceRSWindows::getDepthFrame1Chanels() {
  return _imgdepth1c;
}

const rscg::Intrinsics&
CameraDeviceRSWindows::intrinsics() {
  return _intri;
}

CameraDeviceKinect::CameraDeviceKinect()
  : _paReaderFrame(nullptr),
    _skellTracked(false),
    _paSensor(nullptr),
    _paBodyFrameReader(nullptr),
    _paCoordinateMapper(nullptr),
    _imgdepth{424, 512, CV_16UC1, cv::Scalar(0)}, 
    _imgdepth3{424, 512, CV_8UC3, cv::Scalar(0,0,0)},
    _imdepth1caux{424, 512, CV_8UC1, cv::Scalar(0)},
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
  IBodyFrameSource* paBodyFrameSource = nullptr;
  IDepthFrameSource *paDepthFrameSrc = nullptr;
  hr = _paSensor->Open();
  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_DepthFrameSource(&paDepthFrameSrc);
  }

  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_CoordinateMapper(&_paCoordinateMapper);
  }
  
  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_BodyFrameSource(&paBodyFrameSource);
  }

  if(SUCCEEDED(hr)) {
    hr = paDepthFrameSrc->OpenReader(&_paReaderFrame);
  }

  if(SUCCEEDED(hr)) {
    hr = paBodyFrameSource->OpenReader(&_paBodyFrameReader);
  }

  if(paDepthFrameSrc != nullptr) {
    paDepthFrameSrc->Release();
  }

  if(paBodyFrameSource != nullptr) {
    paBodyFrameSource->Release();
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
  if(_paReaderFrame != nullptr) {
    _paReaderFrame->Release();
  }

  if(_paBodyFrameReader != nullptr) {
    _paBodyFrameReader->Release();
  }

  if(_paCoordinateMapper!= nullptr) {
    _paCoordinateMapper->Release();
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
  if(_paReaderFrame == nullptr) {
    return;
  }

  IDepthFrame *paFrame;
  HRESULT hr = _paReaderFrame->AcquireLatestFrame(&paFrame);
  bool notGet = FAILED(hr);
  while(notGet) {
    hr = _paReaderFrame->AcquireLatestFrame(&paFrame);
    notGet = FAILED(hr);
  }

  if(SUCCEEDED(hr)) {
    uint32_t bufferSize;
    uint16_t *imageArray;
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
      hr = paFrame->AccessUnderlyingBuffer(&bufferSize, &imageArray);
    }

    using cvpoint3u8 = cv::Point3_<uint8_t>;
    const uint16_t maxv = 256;
    uint16_t minRng, maxRng;
    paFrame->get_DepthMinReliableDistance(&minRng);
    paFrame->get_DepthMaxReliableDistance(&maxRng);
    if(SUCCEEDED(hr)) {
      cv::Point pIt{0, 0};
      for(int y = 0; y < nHeigth; ++y) {
        pIt.y = y;
        for(int x = 0; x < nWidth; ++x) {
          uint16_t depth = imageArray[x + y * nWidth];
          uint8_t depthInter = 255 * (depth - 200) / 4800;
          pIt.x = x;
          _imdepth1caux.at<uint8_t>(pIt) = {depthInter};
          _imgdepth.at<uint16_t>(pIt) = {depth};
        }
      }
    }
  
    cv::cvtColor(_imdepth1caux, _imgdepth3, cv::COLOR_GRAY2BGR);
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
CameraDeviceKinect::fetchSkeleton() {
  IBodyFrame* pBodyFrame = NULL;

  _skellTracked = false;
  HRESULT hr = _paBodyFrameReader->AcquireLatestFrame(&pBodyFrame);
  bool notGet = FAILED(hr);
  while(notGet) {
    hr = _paBodyFrameReader->AcquireLatestFrame(&pBodyFrame);
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
            hr = pBody->GetJoints(_joints.size(), _joints.data());
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
    int thick = jTrack ? 3 : 1;

    cv::line(_imgdepth3, screenCoord1, screenCoord2, colorBone, thick);
  }
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
  return _imgdepth3;
}

const cv::Mat&
CameraDeviceKinect::getDepthFrame1Chanels() {
  return _imgdepth;
}

const rscg::Intrinsics&
CameraDeviceKinect::intrinsics() {
  return _intri;
}