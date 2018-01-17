#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <SFML/System/Clock.hpp>

namespace rscg
{

class GraphProbs
{
public:
  GraphProbs(size_t ptsCont);

  void update(const std::vector<float> &f);

  cv::Mat render();

private:
  std::vector<std::vector<float>> _graphPoints;
  int _currInitial;
  std::vector<cv::Scalar> _colorMap;
};

}