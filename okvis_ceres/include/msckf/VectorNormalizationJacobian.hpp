#ifndef INCLUDE_MSCKF_VECTOR_NORMALIZATION_JACOBIAN_HPP_
#define INCLUDE_MSCKF_VECTOR_NORMALIZATION_JACOBIAN_HPP_

#include <glog/logging.h>
#include <Eigen/Dense>

namespace msckf {
class VectorNormalizationJacobian
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  VectorNormalizationJacobian(const Eigen::Vector3d& vecIn) : vecIn_(vecIn) {

  }

  template<typename MatrixType>
  void dxi_dvec(MatrixType* j) const {
    normalizationJacobian(vecIn_, j);
  }

  Eigen::Vector3d normalized() const {
    return vecIn_.normalized();
  }

  template<typename MatrixType>
  static void normalizationJacobian(const Eigen::Vector3d& vecIn, MatrixType* j) {
    double norm = vecIn.norm();
    CHECK_GT(norm, 1e-6);
    double invNorm = 1.0 / norm;
    double invNorm3 = invNorm * invNorm * invNorm;
    *j = Eigen::Matrix3d::Identity() * invNorm - vecIn * vecIn.transpose() * invNorm3;
  }

private:
  Eigen::Vector3d vecIn_; // vector before normalization

};
} // namespace msckf
#endif // INCLUDE_MSCKF_VECTOR_NORMALIZATION_JACOBIAN_HPP_
