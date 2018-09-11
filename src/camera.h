#ifndef __RSCG_CAMERA_HH__
#define __RSCG_CAMERA_HH__

#include <vector>
#include <array>

#include <pxcsensemanager.h>
#include <opencv2/core.hpp>
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
  void fetchSkeleton();

  const cv::Mat& getDepthFrame1Chanels();
  const cv::Mat& getDepthFrame3Chanels();

  const std::vector<Joint>& getSkeletonJointVec();
  const int64_t getCurrentTimeSkeletonFrame();

  void renderSkeletonJointsToDepth();

  const cv::Point2f
  worldToScreenPoint(const CameraSpacePoint& bodyPoint, 
                     const cv::Size windowSize);

  const rscg::Intrinsics& intrinsics() override;
private:
  cv::Mat _imgdepth3, _imgdepth, _screenSkell;
  int64_t _currentTime;
  bool _skellTracked;
  std::vector<Joint> _joints;

  using bone = std::pair<JointType, JointType>;
  const std::vector<bone> _bones;

  IKinectSensor *_paSensor;
  IBodyFrameReader* _paBodyFrameReader;
  ICoordinateMapper* _paCoordinateMapper;
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