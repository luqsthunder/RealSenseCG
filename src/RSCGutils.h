#ifndef REALSENSECG_RSCGUTILS_H
#define REALSENSECG_RSCGUTILS_H

namespace rscg
{

constexpr float MaxCameraDepthM = 2.041f;
constexpr float MinCameraDepthM = 0.011f;


struct Intrinsics
{
  float ppy, ppx, fy, fx;
};

}

#endif //REALSENSECG_RSCGUTILS_H
