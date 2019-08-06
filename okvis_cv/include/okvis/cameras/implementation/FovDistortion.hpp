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
  const Eigen::Vector2d& y = pointUndistorted;
  const double r_u = y.norm();
  //  const double r_u_cubed = r_u * r_u * r_u;
  const double tanwhalf = tan(w_ / 2.);
  //  const double tanwhalfsq = tanwhalf * tanwhalf;
  const double atan_wrd = atan(2. * tanwhalf * r_u);
  double r_rd;

  if (w_ * w_ < 1e-5) {
    // Limit w_ > 0.
    r_rd = 1.0;
  } else {
    if (r_u * r_u < 1e-5) {
      // Limit r_u > 0.
      r_rd = 2. * tanwhalf / w_;
    } else {
      r_rd = atan_wrd / (r_u * w_);
    }
  }

  *pointDistorted = pointUndistorted * r_rd;

  return true;
}

bool FovDistortion::distort(const Eigen::Vector2d& pointUndistorted,
                            Eigen::Vector2d* pointDistorted,
                            Eigen::Matrix2d* pointJacobian,
                            Eigen::Matrix2Xd* parameterJacobian) const {
  const Eigen::Vector2d& y = pointUndistorted;
  const double r_u = y.norm();
  const double r_u_cubed = r_u * r_u * r_u;
  const double tanwhalf = tan(w_ / 2.);
  const double tanwhalfsq = tanwhalf * tanwhalf;
  const double atan_wrd = atan(2. * tanwhalf * r_u);
  double r_rd;

  if (w_ * w_ < 1e-5) {
    // Limit w_ > 0.
    r_rd = 1.0;
  } else {
    if (r_u * r_u < 1e-5) {
      // Limit r_u > 0.
      r_rd = 2. * tanwhalf / w_;
    } else {
      r_rd = atan_wrd / (r_u * w_);
    }
  }

  const double u = y(0);
  const double v = y(1);

  *pointDistorted = pointUndistorted * r_rd;

  Eigen::Matrix2d& J = *pointJacobian;
  J.setZero();

  if (w_ * w_ < 1e-5) {
    J.setIdentity();
  } else if (r_u * r_u < 1e-5) {
    J.setIdentity();
    // The coordinates get multiplied by an expression not depending on r_u.
    J *= (2. * tanwhalf / w_);
  } else {
    const double duf_du =
        (atan_wrd) / (w_ * r_u) - (u * u * atan_wrd) / (w_ * r_u_cubed) +
        (2 * u * u * tanwhalf) /
            (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1));
    const double duf_dv =
        (2 * u * v * tanwhalf) /
            (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1)) -
        (u * v * atan_wrd) / (w_ * r_u_cubed);
    const double dvf_du =
        (2 * u * v * tanwhalf) /
            (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1)) -
        (u * v * atan_wrd) / (w_ * r_u_cubed);
    const double dvf_dv =
        (atan_wrd) / (w_ * r_u) - (v * v * atan_wrd) / (w_ * r_u_cubed) +
        (2 * v * v * tanwhalf) /
            (w_ * (u * u + v * v) * (4 * tanwhalfsq * (u * u + v * v) + 1));

    J << duf_du, duf_dv, dvf_du, dvf_dv;
  }

  if (parameterJacobian) {
    exit(1);  // "FOV parameter Jacobian for distort not implemented!"
  }

  return true;
}

bool FovDistortion::distortWithExternalParameters(
    const Eigen::Vector2d& pointUndistorted, const Eigen::VectorXd& parameters,
    Eigen::Vector2d* pointDistorted, Eigen::Matrix2d* pointJacobian,
    Eigen::Matrix2Xd* parameterJacobian) const {
  // "FOV distortWithExternalParameters not implemented!"
  /*
    const double k1 = parameters[0];
    const double k2 = parameters[1];
    const double p1 = parameters[2];
    const double p2 = parameters[3];
    // first compute the distorted point
    const double u0 = pointUndistorted[0];
    const double u1 = pointUndistorted[1];
    const double mx_u = u0 * u0;
    const double my_u = u1 * u1;
    const double mxy_u = u0 * u1;
    const double rho_u = mx_u + my_u;
    const double rad_dist_u = k1 * rho_u + k2 * rho_u * rho_u;
    (*pointDistorted)[0] = u0 + u0 * rad_dist_u + 2.0 * p1 * mxy_u
        + p2 * (rho_u + 2.0 * mx_u);
    (*pointDistorted)[1] = u1 + u1 * rad_dist_u + 2.0 * p2 * mxy_u
        + p1 * (rho_u + 2.0 * my_u);

    // next the Jacobian w.r.t. changes on the undistorted point
    Eigen::Matrix2d & J = *pointJacobian;
    J(0, 0) = 1 + rad_dist_u + k1 * 2.0 * mx_u + k2 * rho_u * 4 * mx_u
        + 2.0 * p1 * u1 + 6 * p2 * u0;
    J(1, 0) = k1 * 2.0 * u0 * u1 + k2 * 4 * rho_u * u0 * u1 + p1 * 2.0 * u0
        + 2.0 * p2 * u1;
    J(0, 1) = J(1, 0);
    J(1, 1) = 1 + rad_dist_u + k1 * 2.0 * my_u + k2 * rho_u * 4 * my_u
        + 6 * p1 * u1 + 2.0 * p2 * u0;

    if (parameterJacobian) {
      // the Jacobian w.r.t. intrinsics parameters
      Eigen::Matrix2Xd & J2 = *parameterJacobian;
      J2.resize(2,NumDistortionIntrinsics);
      const double r2 =rho_u;
      const double r4 = r2 * r2;

      //[ u0*(u0^2 + u1^2), u0*(u0^2 + u1^2)^2,       2*u0*u1, 3*u0^2 + u1^2]
      //[ u1*(u0^2 + u1^2), u1*(u0^2 + u1^2)^2, u0^2 + 3*u1^2,       2*u0*u1]

      J2(0, 0) = u0 * r2;
      J2(0, 1) = u0 * r4;
      J2(0, 2) = 2.0 * u0 * u1;
      J2(0, 3) = r2 + 2.0 * u0 * u0;

      J2(1, 0) = u1 * r2;
      J2(1, 1) = u1 * r4;
      J2(1, 2) = r2 + 2.0 * u1 * u1;
      J2(1, 3) = 2.0 * u0 * u1;
    }*/
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

bool FovDistortion::undistort(const Eigen::Vector2d& pointDistorted,
                              Eigen::Vector2d* pointUndistorted,
                              Eigen::Matrix2d* pointJacobian) const {
  /*
    // this is expensive: we solve with Gauss-Newton...
    Eigen::Vector2d x_bar = pointDistorted; // initialise at distorted point
    const int n = 5;  // just 5 iterations max.
    Eigen::Matrix2d E;  // error Jacobian

    bool success = false;
    for (int i = 0; i < n; i++) {

      Eigen::Vector2d x_tmp;

      distort(x_bar, &x_tmp, &E);

      Eigen::Vector2d e(pointDistorted - x_tmp);
      Eigen::Vector2d dx = (E.transpose() * E).inverse() * E.transpose() * e;

      x_bar += dx;

      const double chi2 = e.dot(e);
      if (chi2 < 1e-4) {
        success = true;
      }
      if (chi2 < 1e-15) {
        success = true;
        break;
      }

    }
    *pointUndistorted = x_bar;

    // the Jacobian of the inverse map is simply the inverse Jacobian.
    *pointJacobian = E.inverse();

    return success;*/
  // "FOV undistort with pointJacobian not implemented!"
  return false;
}

}  // namespace cameras
}  // namespace okvis
