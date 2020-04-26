#ifndef INCLUDE_MSCKF_MULTIPLE_TRANSFORM_POINT_JACOBIAN_HPP
#define INCLUDE_MSCKF_MULTIPLE_TRANSFORM_POINT_JACOBIAN_HPP

#include <okvis/kinematics/Transformation.hpp>
#include <msckf/memory.h>

namespace okvis {
struct TransformPointJacobianNode {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TransformPointJacobianNode() {}
  okvis::kinematics::Transformation cumulativeLeftTransform_;
  Eigen::Vector4d point_;
  Eigen::Matrix<double, 4, 6> dpoint_dHeadTransform_;
};

/**
 * @brief The MultipleTransformPointJacobianNode struct
 * For q = T1^{a_1} * T2^{a_2} * ... * Tk^{a_k} * p, a_i = {+1, -1}
 * first node contains
 * I=dq_dq, q, dq_d\delta T1
 * second node contains
 * T1^{a_1} = dq_dq1, q1 = T2^{a_2} * ... * Tk^{a_k} * p, dq1_d\delta T2
 * ...
 * second to last node contains
 * T1^{a_1} * T2^{a_2} * ... * T{k-1}^{a_{k-1}} = dq_dq_{k-1}, q{k-1} = Tk^{a_k} * p, dq_{k-1}_d\delta Tk.
 * last node contains T1^{a_1} * T2^{a_2} * ... * Tk^{a_k} = dq_dqk, qk = p, 0.
 *
 * Error in Ti, \delta Ti is defined by okvis::kinematics::minus and oplus.
 * Error in point is defined as p = \hat{p} + \delta p.
 */
class MultipleTransformPointJacobian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MultipleTransformPointJacobian() {}
  MultipleTransformPointJacobian(
      const AlignedVector<okvis::kinematics::Transformation>& transformList,
      const std::vector<int>& exponentList, const Eigen::Vector4d& point)
      : transformList_(transformList),
        exponentList_(exponentList),
        point_(point) {
    computeJacobians();
  }

  void initialize(
      const AlignedVector<okvis::kinematics::Transformation>& transformList,
      const std::vector<int>& exponentList, const Eigen::Vector4d& point) {
    transformList_ = transformList;
    exponentList_ = exponentList;
    point_ = point;
    computeJacobians();
  }

  Eigen::Vector4d evaluate() const;

  void computeJacobians();

  /**
   * @brief dp_dT
   * @param transformIndex transform index in the transformList.
   * @return
   */
  Eigen::Matrix<double, 4, 6> dp_dT(size_t transformIndex) const {
    return transformJacobianList_[transformIndex].cumulativeLeftTransform_.T() *
           transformJacobianList_[transformIndex].dpoint_dHeadTransform_;
  }

  Eigen::Matrix<double, 4, 4> dp_dpoint() const {
    return transformJacobianList_.back().cumulativeLeftTransform_.T();
  }

 private:
  // input
  AlignedVector<okvis::kinematics::Transformation>
      transformList_;              // T1, T2, ... Tk
  std::vector<int> exponentList_;  // a_1, a_2, ..., a_k
  Eigen::Vector4d point_;

  // output
  // The cumulative transforms are: I, T1^{a_1}, T1^{a_1} * T2^{a_2}, ...,
  // T1^{a_1} * T2^{a_2} *... * Tk^{a_k}
  AlignedVector<TransformPointJacobianNode> transformJacobianList_;
};
} // okvis
#endif // INCLUDE_MSCKF_MULTIPLE_TRANSFORM_POINT_JACOBIAN_HPP
