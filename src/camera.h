#ifndef __RSCG_CAMERA_HH__
#define __RSCG_CAMERA_HH__

#include <vector>
#include <array>

#include <pxcsensemanager.h>

#include "RSCGutils.h"

namespace rscg
{

class CameraDevice
{
public:

  virtual const unsigned char* fetchDepthFrame() = 0;
  virtual void releaseFrame() = 0;
  virtual const rscg::Intrinsics& fetchIntrinsics() = 0;
  virtual const float scale() const = 0;
  virtual const std::array<unsigned, 2>& size() const;

protected:
  std::vector<unsigned char> depthframe;
  std::array<unsigned, 2> msize;
};

class CameraDeviceWindows: public CameraDevice
{
public:
  CameraDeviceWindows();
  ~CameraDeviceWindows();

  const unsigned char* fetchDepthFrame() override;
  void releaseFrame() override;
  const rscg::Intrinsics& fetchIntrinsics() override;
  const float scale() const override;

private:
  PXCSenseManager *sm;
  float camScale;
  PXCImage::ImageData data;
  PXCImage* _image;
};

}

#endif