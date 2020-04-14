#ifndef INCLUDE_OKVIS_INVERSE_TRANSFORM_MULTIPLY_JACOBIAN_HPP
#define INCLUDE_OKVIS_INVERSE_TRANSFORM_MULTIPLY_JACOBIAN_HPP

#include <Eigen/Core>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
// Jacobians for $T_z = T_x^{-1} * T_y$.
class InverseTransformMultiplyJacobian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  InverseTransformMultiplyJacobian() {}
  InverseTransformMultiplyJacobian(
      const okvis::kinematics::Transformation& T_WA,
      const okvis::kinematics::Transformation& T_WB)
      : T_WA_(T_WA.r(), T_WA.q()), T_WB_(T_WB.r(), T_WB.q()) {}
  void initialize(const okvis::kinematics::Transformation& T_WA,
                  const okvis::kinematics::Transformation& T_WB) {
    T_WA_.first = T_WA.r();
    T_WA_.second = T_WA.q();
    T_WB_.first = T_WB.r();
    T_WB_.second = T_WB.q();
  }
  inline okvis::kinematics::Transformation multiplyT() const {
    Eigen::Quaterniond q_AB = T_WA_.second.conjugate() * T_WB_.second;
    Eigen::Vector3d t_AB =
        T_WA_.second.conjugate() * (T_WB_.first - T_WA_.first);
    return okvis::kinematics::Transformation(t_AB, q_AB);
  }

  void dp_dp_WA(Eigen::Matrix3d* j) {
    *j = -T_WA_.second.toRotationMatrix().transpose();
  }

  void dp_dtheta_WA(Eigen::Matrix3d* j) {
    Eigen::Vector3d t_AB =
        T_WA_.second.conjugate() * (T_WB_.first - T_WA_.first);
    *j = okvis::kinematics::crossMx(t_AB) *
         T_WA_.second.toRotationMatrix().transpose();
  }

  void dp_dp_WB(Eigen::Matrix3d* j) {
    *j = T_WA_.second.toRotationMatrix().transpose();
  }
  void dtheta_dtheta_WA(Eigen::Matrix3d* j) {
    *j = -T_WA_.second.toRotationMatrix().transpose();
  }
  void dtheta_dtheta_WB(Eigen::Matrix3d* j) {
    *j = T_WA_.second.toRotationMatrix().transpose();
  }

  void dT_dT_WA(Eigen::Matrix<double, 6, 6>* Jzx) {
    Jzx->topLeftCorner<3, 3>() = -T_WA_.second.toRotationMatrix().transpose();
    Eigen::Vector3d t_AB =
        T_WA_.second.conjugate() * (T_WB_.first - T_WA_.first);
    Jzx->topRightCorner<3, 3>() = okvis::kinematics::crossMx(t_AB) *
        T_WA_.second.toRotationMatrix().transpose();
    Jzx->bottomLeftCorner<3, 3>().setZero();
    Jzx->bottomRightCorner<3, 3>() =  -T_WA_.second.toRotationMatrix().transpose();
  }

  void dT_dT_WB(Eigen::Matrix<double, 6, 6>* Jzy) {
    Jzy->topLeftCorner<3, 3>() = T_WA_.second.toRotationMatrix().transpose();
    Jzy->topRightCorner<3, 3>().setZero();
    Jzy->bottomLeftCorner<3, 3>().setZero();
    Jzy->bottomRightCorner<3, 3>() = T_WA_.second.toRotationMatrix().transpose();
  }

 private:
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_WA_;
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_WB_;
};
} // namespace okvis
#endif // INCLUDE_OKVIS_INVERSE_TRANSFORM_MULTIPLY_JACOBIAN_HPP
