/**
 * @file implementation/FovDistortion.hpp
 * @brief Header implementation file for the FovDistortion class.
 * @author Jianzhu Huai
 */

#include <Eigen/LU>
#include <iostream>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

// The default constructor with all zero ki
FovDistortion::FovDistortion() : w_(0.0) { parameters_[0] = 0.0; }

// Constructor initialising ki
FovDistortion::FovDistortion(double w) {
  parameters_[0] = w;
  w_ = w;
}

bool FovDistortion::setParameters(const Eigen::VectorXd& parameters) {
  if (parameters.size() != NumDistortionIntrinsics) {
    return false;
  }
  parameters_ = parameters;
  w_ = parameters[0];
  return true;
}

bool FovDistortion::distort(const Eigen::Vector2d& pointUndistorted,
                            Eigen::Vector2d* pointDistorted) const {
  const double u = pointUndistorted(0);
  const double v = pointUndistorted(1);
  const double radius2 = u * u + v * v;
  const double radius = std::sqrt(radius2);
  const double tanwhalf = tan(w_ / 2.);
  const double atan_wrd = atan(2. * tanwhalf * radius);
  double omega2 = w_ * w_;
  double factor;
  const double eps = 1e-5;
  if (omega2 < eps) {
    // Limit w_ > 0.
    factor = omega2 / 12 - (omega2 * radius2) / 3 + 1;
  } else {
    if (radius2 < eps) {
      // Limit radius > 0.
      const double tanwhalfsq = tanwhalf * tanwhalf;
      factor = -(2 * tanwhalf * (4 * radius2 * tanwhalfsq - 3)) / (3 * w_);
    } else {
      factor = atan_wrd / (radius * w_);
    }
  }
  *pointDistorted = pointUndistorted * factor;
  return true;
}

bool FovDistortion::distort(const Eigen::Vector2d& pointUndistorted,
                            Eigen::Vector2d* pointDistorted,
                            Eigen::Matrix2d* pointJacobian,
                            Eigen::Matrix2Xd* parameterJacobian) const {
  const double u = pointUndistorted(0);
  const double v = pointUndistorted(1);

  const double radius2 = u * u + v * v;
  const double radius = std::sqrt(radius2);
  const double tanwhalf = tan(w_ / 2.);
  const double tanwhalfsq = tanwhalf * tanwhalf;
  const double atan_wrd = atan(2. * tanwhalf * radius);
  double omega2 = w_ * w_;
  double factor;
  const double eps = 1e-5;
  if (omega2 < eps) {
    // Limit w_ > 0.
    factor = omega2 / 12 - (omega2 * radius2) / 3 + 1;
  } else {
    if (radius2 < eps) {
      // Limit radius > 0.
      factor = -(2 * tanwhalf * (4 * radius2 * tanwhalfsq - 3)) / (3 * w_);
      // factor = 2. * tanwhalf / w_;
    } else {
      factor = atan_wrd / (radius * w_);
    }
  }
  *pointDistorted = pointUndistorted * factor;

  Eigen::Matrix2d& J = *pointJacobian;
  double dfactor_dradius;
  double dradius_du;
  double dradius_dv;
  double dfactor_du;
  double dfactor_dv;

  if (omega2 < eps) {
    dfactor_du = -(2 * omega2 * u) / 3;
    dfactor_dv = -(2 * omega2 * v) / 3;
    J << factor + u * dfactor_du, u * dfactor_dv, v * dfactor_du,
        factor + v * dfactor_dv;

  } else if (radius2 < eps) {
    dfactor_du = -(16 * u * tanwhalfsq * tanwhalf) / (3 * w_);
    dfactor_dv = -(16 * v * tanwhalfsq * tanwhalf) / (3 * w_);
    J << factor + u * dfactor_du, u * dfactor_dv, v * dfactor_du,
        factor + v * dfactor_dv;
  } else {
    dradius_du = u / radius;
    dradius_dv = v / radius;
    dfactor_dradius =
        (2 * tanwhalf) / (w_ * radius * (4 * radius2 * tanwhalfsq + 1)) -
        atan(2 * radius * tanwhalf) / (w_ * radius2);
    J << factor + u * dfactor_dradius * dradius_du,
        u * dfactor_dradius * dradius_dv, v * dfactor_dradius * dradius_du,
        factor + v * dfactor_dradius * dradius_dv;
  }

  if (parameterJacobian) {
    Eigen::Matrix2Xd& Ji = *parameterJacobian;
    Ji.resize(2, NumDistortionIntrinsics);
    double dfactor_domega;
    if (omega2 < eps) {
      dfactor_domega = -(w_ * (4 * radius2 - 1)) / 6;
    } else if (radius2 < eps) {
      dfactor_domega = (tanwhalfsq + 1) / w_ +
                       radius2 * ((8 * tanwhalfsq * tanwhalf) / (3 * omega2) -
                                  (4 * tanwhalfsq * (tanwhalfsq + 1)) / w_) -
                       (2 * tanwhalf) / omega2;
    } else {
      dfactor_domega =
          (tanwhalfsq + 1) / (w_ * (4 * radius2 * tanwhalfsq + 1)) -
          atan(2 * radius * tanwhalf) / (omega2 * radius);
    }
    Ji << u * dfactor_domega, v * dfactor_domega;
  }
  return true;
}

bool FovDistortion::distortWithExternalParameters(
    const Eigen::Vector2d& /*pointUndistorted*/, const Eigen::VectorXd& /*parameters*/,
    Eigen::Vector2d* /*pointDistorted*/, Eigen::Matrix2d* /*pointJacobian*/,
    Eigen::Matrix2Xd* /*parameterJacobian*/) const {
  throw std::runtime_error("FOV distortWithExternalParameters not implemented!");
  return false;
}
bool FovDistortion::undistort(const Eigen::Vector2d& pointDistorted,
                              Eigen::Vector2d* pointUndistorted) const {
  const Eigen::Vector2d& y = pointDistorted;

  double mul2tanwby2 = tan(w_ / 2.0) * 2.0;

  // Calculate distance from point to center.
  double r_d = y.norm();

  if (mul2tanwby2 == 0 || r_d == 0) {
    return false;
  }

  // Calculate undistorted radius of point.
  double r_u;
  if (fabs(r_d * w_) <= kMaxValidAngle) {
    r_u = tan(r_d * w_) / (r_d * mul2tanwby2);
  } else {
    return false;
  }

  *pointUndistorted = pointDistorted * r_u;

  return true;
}

bool FovDistortion::undistort(const Eigen::Vector2d& /*pointDistorted*/,
                              Eigen::Vector2d* /*pointUndistorted*/,
                              Eigen::Matrix2d* /*pointJacobian*/) const {
  throw std::runtime_error("FOV undistort with pointJacobian not implemented!");
  return false;
}

}  // namespace cameras
}  // namespace okvis
