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
  void fetchColorFrame();
  void fetchSkeleton();

  const cv::Mat& getDepthFrame1Chanels();
  const cv::Mat& getDepthFrame3Chanels();

  const cv::Mat& getColorFrame();

  const std::vector<Joint>& getSkeletonJointVec();
  const int64_t getCurrentTimeSkeletonFrame();

  bool isThatJointsTracked(const std::vector<JointType> &j);

  void renderSkeletonJointsToDepth();

  bool allJointsTracked();

  const cv::Point2f
  worldToScreenPoint(const CameraSpacePoint& bodyPoint, 
                     const cv::Size windowSize);

  const rscg::Intrinsics& intrinsics() override;
private:
  cv::Mat _imdepth3, _imdepth, _screenSkell, _imdepth1caux, _imColor;
  int64_t _currentTime;
  bool _skellTracked, _allJointsTracked;
  std::vector<Joint> _joints;

  using bone = std::pair<JointType, JointType>;
  const std::vector<bone> _bones;

  IKinectSensor     *_paSensor;
  IBodyFrameReader  *_paSkellFrameReader;
  ICoordinateMapper *_paCoordinateMapper;
  IDepthFrameReader *_paDepthReaderFrame;
  IColorFrameReader *_paColorFrameReader;

  rscg::Intrinsics _intri;
};

class CameraDeviceRSWindows
: public CameraDevice
{
public:
  CameraDeviceRSWindows();
  ~CameraDeviceRSWindows();

  void fetchDepthFrame() override;

  const cv::Mat& getDepthFrame1Chanels();
  const cv::Mat& getDepthFrame3Chanels();

  const rscg::Intrinsics& intrinsics() override;

private:
  PXCSenseManager *sm;

  cv::Mat _imgdepth3, _imgdepth1c;
  
  rscg::Intrinsics _intri;
};

}

#endif