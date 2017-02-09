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

  status = sm->Init();

  imgdepth.resize(640 * 480 * 4, 0);
  _imgdepth1c.resize(640 * 480);
  auto device = sm->QueryCaptureManager()->QueryDevice();

  auto mirrorMode = PXCCapture::Device::MirrorMode::MIRROR_MODE_HORIZONTAL;
  device->SetMirrorMode(mirrorMode);
  auto focus = device->QueryDepthFocalLength();
  auto pp = device->QueryDepthPrincipalPoint();

  _intri = Intrinsics{pp.x, pp.y, focus.x, focus.y};
}

CameraDeviceWindows::~CameraDeviceWindows()
{
  sm->Release();
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
        /* uint16_t byte1 = (uint16_t)imageArray[pixelPitch * x +
        y*imageData.pitches[0]];
        uint16_t byte2 = (uint16_t)(imageArray[pixelPitch * x +
        y*imageData.pitches[0] + 1]) << 8;

        uint16_t depth = byte1 + byte2;*/

        imgdepth[(x * 3) + (y * (sizeX * 3))] = depth;
        imgdepth[(x * 3) + (y * (sizeX * 3)) + 1] = depth;
        imgdepth[(x * 3) + (y * (sizeX * 3)) + 2] = depth;
        imgdepth[(x * 3) + (y * (sizeX * 3)) + 3] = USHRT_MAX;

        _imgdepth1c[x + (y * sizeX)] = depth;
      }
    }
  }
  image->ReleaseAccess(&imageData);
  sm->ReleaseFrame();
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