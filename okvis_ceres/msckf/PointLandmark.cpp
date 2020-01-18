#include <msckf/PointLandmark.hpp>
#include <msckf/PointLandmarkModels.hpp>
#include <okvis/FrameTypedefs.hpp>

namespace msckf {
PointLandmark::PointLandmark(int modelId) : modelId_(modelId) {
}

std::vector<double> PointLandmark::initializePointLandmark(const okvis::MapPoint& mp) {
//  switch (modelId_) {
//    case InverseDepthParameterization::kModelId:
//      InverseDepthParameterization::initializePointLandmark(PointLandmark);
//      break;
//    case HomogeneousPointParameterization::kModelId:
//      HomogeneousPointParameterization::initializePointLandmark();
//      break;
//    case ParallaxAngleParameterization::kModelId:
//      ParallaxAngleParameterization::initializePointLandmark();
//      break;
//  }
  return std::vector<double>(3, 0);
}
}
