#include "graphprobs.h"

using namespace rscg;

GraphProbs::GraphProbs(size_t ptsCont): _currInitial(0)
{
  _graphPoints.resize(14);
  for(size_t i = 0; i < 14; ++i)
    _graphPoints[i].resize(ptsCont, 0);

  _colorMap =
  {
    cv::Scalar(255, 0, 0),
    cv::Scalar(255 ,128, 0),
    cv::Scalar(204, 204, 0),
    cv::Scalar(102, 204, 0),
    cv::Scalar(0, 204, 0),
    cv::Scalar(51, 255, 153),
    cv::Scalar(102, 255, 255),
    cv::Scalar(102, 102, 255),
    cv::Scalar(255, 102, 255),
    cv::Scalar(153, 51, 255),
    cv::Scalar(255, 0 ,127),
    cv::Scalar(0, 0, 153),
    cv::Scalar(255, 255, 51),
    cv::Scalar(64, 64, 64)
  };
}

void
GraphProbs::update(const std::vector<float> &f)
{
  int n = _graphPoints[0].size();
  _currInitial = (_currInitial + 1) % n;
  for(size_t i = 0; i < 14; ++i)
    _graphPoints[i][_currInitial] = f[i];
}

cv::Mat
GraphProbs::render()
{
  int n = _graphPoints[0].size();
  int x = 0;
  cv::Mat out{480, 640, CV_8UC3, cv::Scalar(0, 0, 0)};

  for(size_t i = 0; i < 14; ++i)
  {
    x = 0;
    for(size_t i2 = 0; i2 < n - 1; ++i2)
    {
      int y = 420 - (int)(_graphPoints[i][(_currInitial + i2 + 1) % n] * 200 + 20);
      int y2 = 420 - (int)(_graphPoints[i][((_currInitial + i2 + 2) % n)] * 200 + 20);
      cv::line(out, {x, y}, {x + (640 / (n - 1)), y2}, _colorMap[i]);
      x += (640 / (n - 1));
    }
  }

  return out;
}