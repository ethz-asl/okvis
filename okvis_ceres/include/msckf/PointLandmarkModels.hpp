#ifndef INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_
#define INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_

namespace msckf {
class HomogeneousPointParameterization
{
public:
  static const int kModelId = 0;
  static void bearingVectorInCamera() {}
  static void bearingVectorInWorld() {}
};

class InverseDepthParameterization
{
public:
  static const int kModelId = 1;
};

class ParallaxAngleParameterization {
public:
  static const int kModelId = 2;
};

// add the model switch functions
} // namespace msckf

#endif // INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_
