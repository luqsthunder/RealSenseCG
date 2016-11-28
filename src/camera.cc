#include "camera.h"

#include <iostream>


using namespace rscg;

const std::array<unsigned, 2>&
CameraDevice::size() const
{
  return msize;
}

CameraDeviceWindows::CameraDeviceWindows()
{
  msize = {640, 480};
  sm = PXCSenseManager::CreateInstance();

  sm->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, 640, 480);

  // Initialize and Stream Samples
  sm->Init();

  depthframe.resize(msize[0] * msize[1]);

  camScale = sm->QueryCaptureManager()->QueryDevice()->QueryDepthUnit();
}

CameraDeviceWindows::~CameraDeviceWindows()
{
  sm->Release();
}

const unsigned char*
CameraDeviceWindows::fetchDepthFrame()
{
  // This function blocks until a sample is ready
  if (sm->AcquireFrame(true)<PXC_STATUS_NO_ERROR) 
    std::cerr << "could not aquire frame" << std::endl;

  auto streamtype = PXCCapture::StreamType::STREAM_TYPE_DEPTH;
  _image = (*sm->QuerySample())[streamtype];

 
  PXCImage::Rotation rotation = _image->QueryRotation();
  pxcStatus sts = _image->AcquireAccess(PXCImage::ACCESS_READ,
                                       PXCImage::PIXEL_FORMAT_DEPTH_RAW,
                                       rotation, PXCImage::OPTION_ANY,
                                       &data);

  return data.planes[0];
}

const rscg::Intrinsics&
CameraDeviceWindows::fetchIntrinsics()
{
  return {0, 0, 0, 0};
}

const float
CameraDeviceWindows::scale() const
{
  return camScale;
}

void
CameraDeviceWindows::releaseFrame()
{
  _image->ReleaseAccess(&data);
  sm->ReleaseFrame();
}