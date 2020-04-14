/**
 * @file sophus_operators.hpp
 * @brief File with operators copied from sophus for SO(3).
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_KINEMATICS_SOPHUS_OPERATORS_HPP_
#define INCLUDE_OKVIS_KINEMATICS_SOPHUS_OPERATORS_HPP_

#include <stdint.h>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <okvis/kinematics/Transformation.hpp>
#include <glog/logging.h>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief kinematics Namespace for kinematics functionality, i.e. transformations and stuff.
namespace kinematics {

template <typename Scalar>
struct SophusConstants {
  EIGEN_ALWAYS_INLINE static Scalar epsilon() {
    return static_cast<Scalar>(1e-10);
  }

  EIGEN_ALWAYS_INLINE static Scalar pi() { return static_cast<Scalar>(M_PI); }
};

// from sophus/so3.hpp
template <typename Scalar>
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE static Eigen::Matrix<Scalar, 3, 1> vee(
    const Eigen::Matrix<Scalar, 3, 3>& Omega) {
  return static_cast<Scalar>(0.5) *
         Eigen::Matrix<Scalar, 3, 1>(Omega(2, 1) - Omega(1, 2),
                                     Omega(0, 2) - Omega(2, 0),
                                     Omega(1, 0) - Omega(0, 1));
}

/// Warn: Do not use sinc or its templated version for autodiff involving quaternions
///  as its real part may be calculated without considering the infinisimal input.
/// Use expAndTheta borrowed from Sophus instead for this purpose
template <typename Scalar>
Eigen::Quaternion<Scalar> expAndTheta(const Eigen::Matrix<Scalar, 3, 1> & omega) {
    Scalar theta_sq = omega.squaredNorm();
    Scalar theta = sqrt(theta_sq);
    Scalar half_theta = static_cast<Scalar>(0.5)*(theta);

    Scalar imag_factor;
    Scalar real_factor;
    if(theta<SophusConstants<Scalar>::epsilon()) {
      Scalar theta_po4 = theta_sq*theta_sq;
      imag_factor = static_cast<Scalar>(0.5)
                    - static_cast<Scalar>(1.0/48.0)*theta_sq
                    + static_cast<Scalar>(1.0/3840.0)*theta_po4;
      real_factor = static_cast<Scalar>(1)
                    - static_cast<Scalar>(0.5)*theta_sq +
                    static_cast<Scalar>(1.0/384.0)*theta_po4;
    } else {
      Scalar sin_half_theta = sin(half_theta);
      imag_factor = sin_half_theta/theta;
      real_factor = cos(half_theta);
    }

    return Eigen::Quaternion<Scalar>(real_factor,
                                               imag_factor*omega.x(),
                                               imag_factor*omega.y(),
                                               imag_factor*omega.z());
}

// From sophus so3.hpp
template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> logAndTheta(const Eigen::Quaternion<Scalar> & other,
                          Scalar * theta) {
  Scalar squared_n
      = other.vec().squaredNorm();
  Scalar n = sqrt(squared_n);
  Scalar w = other.w();

  Scalar two_atan_nbyw_by_n;

  // Atan-based log thanks to
  //
  // C. Hertzberg et al.:
  // "Integrating Generic Sensor Fusion Algorithms with Sound State
  // Representation through Encapsulation of Manifolds"
  // Information Fusion, 2011

  if (n < SophusConstants<Scalar>::epsilon()) {
    // If quaternion is normalized and n=0, then w should be 1;
    // w=0 should never happen here!
    CHECK_GT(abs(w), SophusConstants<Scalar>::epsilon()) <<
                  "Quaternion should be normalized!";
    Scalar squared_w = w*w;
    two_atan_nbyw_by_n = static_cast<Scalar>(2) / w
                         - static_cast<Scalar>(2)*(squared_n)/(w*squared_w);
  } else {
    if (abs(w)<SophusConstants<Scalar>::epsilon()) {
      if (w > static_cast<Scalar>(0)) {
        two_atan_nbyw_by_n = M_PI/n;
      } else {
        two_atan_nbyw_by_n = -M_PI/n;
      }
    }else{
      two_atan_nbyw_by_n = static_cast<Scalar>(2) * atan(n/w) / n;
    }
  }

  *theta = two_atan_nbyw_by_n*n;

  return two_atan_nbyw_by_n * other.vec();
}
/**
 * @brief ominus The inverse of Tbar = T.oplus(delta).
 * @param Tbar
 * @param T
 * @return delta
 */
inline Eigen::Matrix<double, 6, 1> ominus(
    const okvis::kinematics::Transformation& Tbar,
    const okvis::kinematics::Transformation& T) {
  Eigen::Matrix<double, 3, 3> dR = Tbar.C() * T.C().transpose();
  Eigen::Matrix<double, 6, 1> delta;
  delta.head<3>() = Tbar.r() - T.r();
  delta.tail<3>() = vee(dR);
  return delta;
}
} // namespace kinematics
} // namespace okvis

#endif /* INCLUDE_OKVIS_KINEMATICS_SOPHUS_OPERATORS_HPP_ */
