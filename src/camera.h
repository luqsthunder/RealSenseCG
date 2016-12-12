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

  virtual const std::vector<uint8_t>& fetchDepthFrame() = 0;
  virtual const rscg::Intrinsics& intrinsics() = 0;

protected:
  std::array<unsigned, 2> msize;
};

class CameraDeviceWindows: public CameraDevice
{
public:
  CameraDeviceWindows();
  ~CameraDeviceWindows();

  const std::vector<uint8_t>& fetchDepthFrame() override;
  const rscg::Intrinsics& intrinsics() override;

private:
  PXCSenseManager *sm;

  std::vector<uint8_t> imgdepth;
  rscg::Intrinsics _intri;
};

}

#endif