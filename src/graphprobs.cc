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
    cv::Scalar(0 ,255, 0),
    cv::Scalar(0, 0, 255),
    cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255),
    cv::Scalar(192, 192, 192),
    cv::Scalar(128, 0, 0),
    cv::Scalar(128, 128, 0),
    cv::Scalar(0, 128, 0),
    cv::Scalar(128, 0, 128),
    cv::Scalar(0, 128 ,128),
    cv::Scalar(0, 0, 128),
    cv::Scalar(139, 69, 19),
    cv::Scalar(255, 255, 255)
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
  cv::Mat out{480, 3*640, CV_8UC3, cv::Scalar(0, 0, 0)};

  int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
  double fontScale = 1;
  int thickness = 3;
  int baseline;

  std::string text;

  for(size_t i = 0; i < 14; ++i)
  {
    baseline = 0;
    text = std::to_string(i);
    cv::Size textSize = cv::getTextSize(text, fontFace,
                                        fontScale, thickness, &baseline);
    auto v = (int)(_graphPoints[i][_currInitial] * 200 + 20);
    baseline += 1;
    cv::Point textOrg(11 * i, 479 - v);

    cv::putText(out, text, textOrg, fontFace, fontScale,
                _colorMap[i], 1, 1);

    x = 0;
    for(size_t i2 = 0; i2 < n - 1; ++i2)
    {
      int y = 479 - (int)(_graphPoints[i][(_currInitial + i2 + 1) % n] * 200 + 20);
      int y2 = 479 - (int)(_graphPoints[i][((_currInitial + i2 + 2) % n)] * 200 + 20);
      cv::line(out, {x, y}, {x + (3*640 / (n - 1)), y2}, _colorMap[i]);
      x += (3*640 / (n - 1));
    }
  }

  return out;
}