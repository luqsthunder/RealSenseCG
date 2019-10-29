#ifndef __RSCG_CAMERA_HH__
#define __RSCG_CAMERA_HH__

#include <vector>
#include <array>

#include <opencv2/core.hpp>
#include <Kinect.h>

namespace rscg
{

class CameraDevice
{
public:
    virtual void fetchDepthFrame() = 0;
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
};
}

#endif
