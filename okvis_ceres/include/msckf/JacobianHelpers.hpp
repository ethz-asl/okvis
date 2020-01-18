
/**
 * @file JacobianHelpers.hpp
 * @brief Header file for utility functions to help computing Jacobians.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_MSCKF_JACOBIAN_HELPERS_HPP_
#define INCLUDE_MSCKF_JACOBIAN_HELPERS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace okvis {
namespace ceres {

template <typename Scalar>
struct SophusConstants {
  EIGEN_ALWAYS_INLINE static Scalar epsilon() {
    return static_cast<Scalar>(1e-10);
  }

  EIGEN_ALWAYS_INLINE static Scalar pi() { return static_cast<Scalar>(M_PI); }
};

/// Warn: Do not use sinc or its templated version for autodiff involving quaternions
///  as its real part may be calculated without considering the infinisimal input.
/// Use expAndTheta borrowed from Sophus instead for this purpose
template <typename Scalar>
Eigen::Quaternion<Scalar> expAndTheta(const Eigen::Matrix<Scalar, 3, 1> & omega) {
    Scalar theta_sq = omega.squaredNorm();
    Scalar theta = sqrt(theta_sq);
    Scalar half_theta = static_cast<Scalar>(0.5)*(theta);

    Scalar imag_factor;
    Scalar real_factor;;
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

template <int globalDim, int localDim, int numResiduals>
inline void zeroJacobian(int index, double** jacobians, double** jacobiansMinimal) {
  using JacType = typename std::conditional<
      (globalDim > 1),
      Eigen::Matrix<double, numResiduals, globalDim, Eigen::RowMajor>,
      Eigen::Matrix<double, numResiduals, globalDim> >::type;
  using MinimalJacType = typename std::conditional<
      (localDim > 1),
      Eigen::Matrix<double, numResiduals, localDim, Eigen::RowMajor>,
      Eigen::Matrix<double, numResiduals, localDim> >::type;
  if (jacobians[index] != NULL) {
    Eigen::Map<JacType> J0(jacobians[index]);
    J0.setZero();
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[index] != NULL) {
        Eigen::Map<MinimalJacType> J0_minimal_mapped(jacobiansMinimal[index]);
        J0_minimal_mapped.setZero();
      }
    }
  }
}

}  // namespace ceres
}  // namespace okvis
#endif // INCLUDE_MSCKF_JACOBIAN_HELPERS_HPP_
