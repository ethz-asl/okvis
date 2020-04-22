
/**
 * @file JacobianHelpers.hpp
 * @brief Header file for utility functions to help computing Jacobians.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_MSCKF_JACOBIAN_HELPERS_HPP_
#define INCLUDE_MSCKF_JACOBIAN_HELPERS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
namespace ceres {
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
