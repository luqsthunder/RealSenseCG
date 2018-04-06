#include "rect.h"

using namespace rscg;

rscg::Rect<int>
rscg::boundingSquare(const cv::Mat &im) {
  rscg::Rect<int> bounds;
  int xl, yl;
  std::pair<int, int> rngX, rngY, rangeXaux, rangeYaux;

  rngX = rangeXaux = {1000, -1};
  int dist1, dist2;
  for(int y = 0; y < 640; ++y) {
    for(int x = 0; x < 480; ++x) {
      if(im.at<uint8_t>(x, y) > 0) {
        if(rngX.first > x)
          rngX.first = x;
        if(rngX.second < x)
          rngX.second = x;
      }
    }
  }

  rngY = rangeYaux = {1000, -1};
  for(int x = 0; x < 480; ++x) {
    for(int y = 0; y < 640; ++y) {
      if(im.at<uint8_t>(x, y) > 0) {
        if(rngY.first > y) {
          rngY.first = y;
        }
        if(rngY.second < y) {
          rngY.second = y;
        }
      }
    }
  }

  dist1 = rngY.second - rngY.first;
  dist2 = rngX.second - rngX.first;

  int max = (dist1 > dist2) ? dist1 : dist2;

  bounds.y = (dist1 < dist2) ? rngX.first : rngX.first - ((max - dist2) / 2);
  bounds.x = (dist1 < dist2) ? rngY.first - ((max - dist1) / 2) : rngY.first;
  bounds.h = bounds.w = max;
  
  return bounds;
}
