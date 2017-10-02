#pragma once

#include <map>
#include <opencv2/opencv.hpp>

namespace rscg
{
template<typename T>
struct Rect
{
  Rect(): Rect(0, 0, 0, 0) { }
  Rect(T x, T y, T w, T h): x(x), y(y), w(w), h(h) { }

  T x, y, w, h;
};

rscg::Rect<int>
boundingSquare(const cv::Mat &im);

}