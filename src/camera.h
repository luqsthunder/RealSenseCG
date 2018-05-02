#ifndef __RSCG_CAMERA_HH__
#define __RSCG_CAMERA_HH__

#include <vector>
#include <array>

#include <pxcsensemanager.h>

#include <Kinect.h>

#include "RSCGutils.h"

namespace rscg
{

class CameraDevice
{
public:
  virtual void fetchDepthFrame() = 0;

  virtual const rscg::Intrinsics& intrinsics() = 0;

protected:
  std::array<unsigned, 2> msize;
};


class CameraDeviceKinect : public CameraDevice
{
public:
  CameraDeviceKinect();
  ~CameraDeviceKinect();

  void fetchDepthFrame() override;

  const std::vector<uint16_t>& getDepthFrame1Chanels();
  const std::vector<uint16_t>& getDepthFrame3Chanels();

  const rscg::Intrinsics& intrinsics() override;
private:
  std::vector<uint16_t> imgdepth3;
  std::vector<uint16_t> imgdepth;

  IKinectSensor *_paSensor;
  IDepthFrameReader *_paReaderFrame;
  rscg::Intrinsics _intri;
};

class CameraDeviceRSWindows
: public CameraDevice
{
public:
  CameraDeviceRSWindows();
  ~CameraDeviceRSWindows();

  void fetchDepthFrame() override;

  const std::vector<uint16_t>& getDepthFrame1Chanels();
  const std::vector<uint16_t>& getDepthFrame3Chanels();

  const rscg::Intrinsics& intrinsics() override;

private:
  PXCSenseManager *sm;

  std::vector<uint16_t> imgdepth3;
  std::vector<uint16_t> _imgdepth1c;
  
  rscg::Intrinsics _intri;
};

}

#endif