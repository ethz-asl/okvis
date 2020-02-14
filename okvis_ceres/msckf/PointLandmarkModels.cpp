#include "msckf/PointLandmarkModels.hpp"
#include <msckf/ParallaxAnglePoint.hpp>

namespace msckf {
bool ParallaxAngleParameterization::plus(const double* x,
                                                const double* delta,
                                                double* x_plus_delta) {
  Eigen::Map<const Eigen::Vector3d> _delta(delta);
  LWF::ParallaxAnglePoint pap(x[3], x[0], x[1], x[2], x[4], x[5]);
  pap.boxPlus(_delta, pap);

  const double* bearingData = pap.n_.data();
  x_plus_delta[0] = bearingData[0];
  x_plus_delta[1] = bearingData[1];
  x_plus_delta[2] = bearingData[2];
  x_plus_delta[3] = bearingData[3];
  const double* thetaData = pap.theta_.data();
  x_plus_delta[4] = thetaData[0];
  x_plus_delta[5] = thetaData[1];
  return true;
}
} // namespace msckf
