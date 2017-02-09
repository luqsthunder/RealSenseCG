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
  virtual void fetchDepthFrame() = 0;
  virtual const std::vector<uint16_t>& getDepthFrame4Chanels() = 0;
  virtual const std::vector<uint16_t>& getDepthFrame1Chanels() = 0;
  virtual const rscg::Intrinsics& intrinsics() = 0;

protected:
  std::array<unsigned, 2> msize;
};

class CameraDeviceWindows: public CameraDevice
{
public:
  CameraDeviceWindows();
  ~CameraDeviceWindows();

  void fetchDepthFrame() override;

  const std::vector<uint16_t>& getDepthFrame4Chanels() override;
  const std::vector<uint16_t>& getDepthFrame1Chanels() override;
  const rscg::Intrinsics& intrinsics() override;

private:
  PXCSenseManager *sm;

  std::vector<uint16_t> imgdepth;
  std::vector<uint16_t> _imgdepth1c;
  rscg::Intrinsics _intri;
};

}

#endif