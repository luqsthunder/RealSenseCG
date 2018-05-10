#include "camera.h"

#include <iostream>
#include <climits>
#include <strsafe.h>


using namespace rscg;

CameraDeviceRSWindows::CameraDeviceRSWindows() : sm(nullptr) {
  pxcStatus status;

  msize = {640, 480};
  sm = PXCSenseManager::CreateInstance();

  if(!sm) {
    throw std::runtime_error("could not create camera manager");
  }

  sm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, 640, 480, 60);
  sm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, 640, 480, 60);
  status = sm->Init();

  _imgdepth1c.resize(640 * 480);
  auto device = sm->QueryCaptureManager()->QueryDevice();

  auto mirrorMode = PXCCapture::Device::MirrorMode::MIRROR_MODE_HORIZONTAL;
  device->SetMirrorMode(mirrorMode);
  auto focus = device->QueryDepthFocalLength();
  auto pp = device->QueryDepthPrincipalPoint();

  _intri = Intrinsics{pp.x, pp.y, focus.x, focus.y};
  imgdepth3.resize(640 * 480 * 3);
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

        imgdepth3[(x * 3) + (y * (sizeX * 3))] = depth;
        imgdepth3[(x * 3) + (y * (sizeX * 3)) + 1] = depth;
        imgdepth3[(x * 3) + (y * (sizeX * 3)) + 2] = depth;

        _imgdepth1c[x + (y * sizeX)] = depth;
      }
    }
  }
  image->ReleaseAccess(&imageData);
  sm->ReleaseFrame();
}

const std::vector<uint16_t>&
CameraDeviceRSWindows::getDepthFrame3Chanels() {
  return imgdepth3;
}

const std::vector<uint16_t>&
CameraDeviceRSWindows::getDepthFrame1Chanels() {
  return _imgdepth1c;
}

const rscg::Intrinsics&
CameraDeviceRSWindows::intrinsics() {
  return _intri;
}

CameraDeviceKinect::CameraDeviceKinect(): _paReaderFrame(nullptr) {
  HRESULT hr = GetDefaultKinectSensor(&_paSensor);
  if(FAILED(hr)) {
    std::cerr << "error on initializing kinect \n";
    return;
  }

  IDepthFrameSource *paDepthFrameSrc = nullptr;
  hr = _paSensor->Open();
  if(SUCCEEDED(hr)) {
    hr = _paSensor->get_DepthFrameSource(&paDepthFrameSrc);
  }

  if(SUCCEEDED(hr)) {
    hr = paDepthFrameSrc->OpenReader(&_paReaderFrame);
  }

  if(paDepthFrameSrc != nullptr) {
    paDepthFrameSrc->Release();
  }

  if(FAILED(hr) || _paSensor == nullptr) {
    std::cerr << "no ready kinect found \n";
    return;
  }

  imgdepth.resize(512 * 424);
  imgdepth3.resize(512 * 424 * 3);
}

CameraDeviceKinect::~CameraDeviceKinect()
{
  if(_paReaderFrame != nullptr)
  {
    _paReaderFrame->Release();
  }

  if(_paSensor != nullptr)
  {
    _paSensor->Close();
    _paSensor->Release();
  }
}

void
CameraDeviceKinect::fetchDepthFrame()
{
  if(_paReaderFrame == nullptr) {
    return;
  }

  IDepthFrame *paFrame;
  HRESULT hr = _paReaderFrame->AcquireLatestFrame(&paFrame);
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

    if(SUCCEEDED(hr)) {
      for(int y = 0; y < nHeigth; ++y) {
        for(int x = 0; x < nWidth; ++x) {
          uint16_t depth = imageArray[x + y * nWidth];

          imgdepth3[(x * 3) + (y * (nWidth * 3))] = depth;
          imgdepth3[(x * 3) + (y * (nWidth * 3)) + 1] = depth;
          imgdepth3[(x * 3) + (y * (nWidth * 3)) + 2] = depth;

          imgdepth[x + (y * nWidth)] = depth;
        }
      }
    }
  
    paFrameDescription->Release();
  }

  if(paFrame != nullptr) {
    paFrame->Release();
  }
}

const std::vector<uint16_t>&
CameraDeviceKinect::getDepthFrame3Chanels()
{
  return imgdepth3;
}

const std::vector<uint16_t>&
CameraDeviceKinect::getDepthFrame1Chanels()
{
  return imgdepth;
}

const rscg::Intrinsics&
CameraDeviceKinect::intrinsics()
{
  return _intri;
}