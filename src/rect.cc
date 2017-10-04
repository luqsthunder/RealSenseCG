#include "rect.h"

using namespace rscg;

rscg::Rect<int>
rscg::boundingSquare(const cv::Mat &im)
{
  rscg::Rect<int> bounds;
  int xl, yl;
  std::pair<int, int> rangeX, rangeY, rangeXaux, rangeYaux;

  rangeX = rangeXaux = {1000, -1};
  int dist1, dist2;
  for(int y = 0; y < 640; ++y)
  {
    for(int x = 0; x < 480; ++x)
    {
      if(im.at<float>(x, y) > 0)
      {
        if(rangeX.first > x)
          rangeX.first = x;
        if(rangeX.second < x)
          rangeX.second = x;
      }
    }
  }

  rangeY = rangeYaux = {1000, -1};
  for(int x = 0; x < 480; ++x)
  {
    for(int y = 0; y < 640; ++y)
    {
      if(im.at<float>(x, y) > 0)
      {
        if(rangeY.first > y)
          rangeY.first = y;
        if(rangeY.second < y)
          rangeY.second = y;
      }
    }
  }

  dist1 = rangeY.second - rangeY.first;
  dist2 = rangeX.second - rangeX.first;

  bounds.y = rangeX.first;
  bounds.x = rangeY.first;
  bounds.w = (dist1 > dist2) ? dist1 : dist2;
  bounds.h = bounds.w;

  return bounds;
}