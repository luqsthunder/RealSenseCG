#include "camera.h"

#include <iostream>
#include <climits>


using namespace rscg;

CameraDeviceWindows::CameraDeviceWindows() : sm(nullptr)
{
  pxcStatus status;

  msize = {640, 480};
  sm = PXCSenseManager::CreateInstance();

  if(!sm)
    throw std::runtime_error("could not create camera manager");

  sm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, 640, 480, 60);
  sm->EnableStream(PXCCapture::STREAM_TYPE_COLOR, 640, 480, 60);
  status = sm->Init();

  imgdepth.resize(640 * 480 * 4, 0);
  _imgdepth1c.resize(640 * 480);
  auto device = sm->QueryCaptureManager()->QueryDevice();

  auto mirrorMode = PXCCapture::Device::MirrorMode::MIRROR_MODE_HORIZONTAL;
  device->SetMirrorMode(mirrorMode);
  auto focus = device->QueryDepthFocalLength();
  auto pp = device->QueryDepthPrincipalPoint();

  _intri = Intrinsics{pp.x, pp.y, focus.x, focus.y};
  imgdepth3.resize(640 * 480 * 3);

  _colorim.resize(3 * 480 * 640);
}

CameraDeviceWindows::~CameraDeviceWindows()
{
  sm->Release();
}

void
CameraDeviceWindows::fetchColorFrame()
{
  /// This function blocks until a sample is ready
  if(sm->AcquireFrame(true) < PXC_STATUS_NO_ERROR)
    std::cerr << "could not aquire frame" << std::endl;

  PXCCapture::Sample *sample = sm->QuerySample();
  auto image = sample->color;

  PXCImage::ImageData imageData;
  PXCImage::ImageInfo imageInfo = image->QueryInfo();

  /// Below is a common AcquireAccess - Process - ReleaseAccess pattern
  /// for RealSense images.
  pxcStatus status;

  pxcI32 sizeX = imageInfo.width;  // Number of pixels in X
  pxcI32 sizeY = imageInfo.height; // Number of pixels in Y

  status = image->AcquireAccess(PXCImage::ACCESS_READ, 
                                PXCImage::PixelFormat::PIXEL_FORMAT_RGB24,
                                &imageData);

 /* if(imageData.pitches[0] != 2 * sizeX)
    throw std::runtime_error("Unexpected data in buffer");
*/

  if(status == PXC_STATUS_NO_ERROR)
  {
    /// becouse every pixel is store as 2 bytes, and obviously pitch is 2
    /// times width, changing array pointer to a double size will work 
    /// equals as manipulating bytes
    uint8_t* imre = (uint8_t*)imageData.planes[0];

    for(pxcI32 y = 0; y < sizeY; y++)
    {
      for(pxcI32 x = 0; x < sizeX; x++)
      {
        _colorim[3 * (x + y * 640)]     = imre[3 * (x + y * 640)];
        _colorim[3 * (x + y * 640) + 1] = imre[3 * (x + y * 640) + 1];
        _colorim[3 * (x + y * 640) + 2] = imre[3 * (x + y * 640) + 2];
      }
    }
  }
  image->ReleaseAccess(&imageData);
  sm->ReleaseFrame();
}


void 
CameraDeviceWindows::fetchDepthFrame()
{
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

  if(imageData.pitches[0] != 2 * sizeX)
    throw std::runtime_error("Unexpected data in buffer");

  const pxcI32 pixelPitch = 2;

  if(status == PXC_STATUS_NO_ERROR)
  {
    /// becouse every pixel is store as 2 bytes, and obviously pitch is 2
    /// times width, changing array pointer to a double size will work 
    /// equals as manipulating bytes

    //pxcU16* imageArray = (pxcU16*)imageData.planes[0];
    uint16_t* imageArray = (uint16_t*)imageData.planes[0];
    //pxcBYTE* imageArray = imageData.planes[0];

    for(pxcI32 y = 0; y < sizeY; y++)
    {
      for(pxcI32 x = 0; x < sizeX; x++)
      {

        uint16_t depth = imageArray[x + y * sizeX];

        imgdepth3[(x * 3) + (y * (sizeX * 3))] = depth;
        imgdepth3[(x * 3) + (y * (sizeX * 3)) + 1] = depth;
        imgdepth3[(x * 3) + (y * (sizeX * 3)) + 2] = depth;

        imgdepth[(x * 4) + (y * (sizeX * 4))] = depth;
        imgdepth[(x * 4) + (y * (sizeX * 4)) + 1] = depth;
        imgdepth[(x * 4) + (y * (sizeX * 4)) + 2] = depth;
        imgdepth[(x * 4) + (y * (sizeX * 4)) + 3] = USHRT_MAX;

        _imgdepth1c[x + (y * sizeX)] = depth;
      }
    }
  }
  image->ReleaseAccess(&imageData);
  sm->ReleaseFrame();
}

const std::vector<uint16_t>&
CameraDeviceWindows::getDepthFrame3Chanels()
{
  return imgdepth3;
}

const std::vector<uint8_t>&
CameraDeviceWindows::getColorFrame()
{
  return _colorim;
}

const std::vector<uint16_t>&
CameraDeviceWindows::getDepthFrame4Chanels()
{
  return imgdepth;
}

const std::vector<uint16_t>&
CameraDeviceWindows::getDepthFrame1Chanels()
{
  return _imgdepth1c;
}

const rscg::Intrinsics&
CameraDeviceWindows::intrinsics()
{
  return _intri;
}

CameraDeviceKinect::CameraDeviceKinect()
{
}

void
CameraDeviceKinect::fetchColorFrame()
{
}

void
CameraDeviceKinect::fetchDepthFrame()
{
}

const std::vector<uint16_t>&
CameraDeviceKinect::getDepthFrame3Chanels()
{
  return imgdepth3;
}

const std::vector<uint8_t>&
CameraDeviceKinect::getColorFrame()
{
  return _colorim;
}

const std::vector<uint16_t>&
CameraDeviceKinect::getDepthFrame4Chanels()
{
  return imgdepth;
}

const std::vector<uint16_t>&
CameraDeviceKinect::getDepthFrame1Chanels()
{
  return _imgdepth1c;
}

const rscg::Intrinsics&
CameraDeviceKinect::intrinsics()
{
  return _intri;
}