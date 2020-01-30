#ifndef INCLUDE_MSCKF_TRANSFORM_MULTIPLY_JACOBIAN_HPP_
#define INCLUDE_MSCKF_TRANSFORM_MULTIPLY_JACOBIAN_HPP_

#include <okvis/kinematics/Transformation.hpp>

namespace msckf {
class TransformMultiplyJacobian {
 public:
  TransformMultiplyJacobian(const okvis::kinematics::Transformation& T_AB,
                            const okvis::kinematics::Transformation& T_BC)
      : T_AB_(T_AB.r(), T_AB.q()), T_BC_(T_BC.r(), T_BC.q()) {}

  TransformMultiplyJacobian() {}

  void initialize(const okvis::kinematics::Transformation& T_AB,
                  const okvis::kinematics::Transformation& T_BC) {
    T_AB_.first = T_AB.r();
    T_AB_.second = T_AB.q();
    T_BC_.first = T_BC.r();
    T_BC_.second = T_BC.q();
  }

  inline std::pair<Eigen::Vector3d, Eigen::Quaterniond> multiply() const {
    Eigen::Quaterniond q_AC = T_AB_.second * T_BC_.second;
    Eigen::Vector3d t_AC = T_AB_.second * T_BC_.first + T_AB_.first;
    return std::make_pair(t_AC, q_AC);
  }
  inline okvis::kinematics::Transformation multiplyT() const {
    Eigen::Quaterniond q_AC = T_AB_.second * T_BC_.second;
    Eigen::Vector3d t_AC = T_AB_.second * T_BC_.first + T_AB_.first;
    return okvis::kinematics::Transformation(t_AC, q_AC);
  }

  void dtheta_dtheta_AB(Eigen::Matrix3d* j) { j->setIdentity(); }

  void dtheta_dt_AB(Eigen::Matrix3d* j) { j->setZero(); }

  void dtheta_dtheta_BC(Eigen::Matrix3d* j) {
    *j = T_AB_.second.toRotationMatrix();
  }

  void dtheta_dt_BC(Eigen::Matrix3d* j) { j->setZero(); }

  void dp_dtheta_AB(Eigen::Matrix3d* j) {
    *j = okvis::kinematics::crossMx(T_AB_.second * -T_BC_.first);
  }

  void dp_dt_AB(Eigen::Matrix3d* j) { j->setIdentity(); }

  void dp_dtheta_BC(Eigen::Matrix3d* j) { j->setZero(); }

  void dp_dt_BC(Eigen::Matrix3d* j) { *j = T_AB_.second.toRotationMatrix(); }

 private:
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_AB_;
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_BC_;
};
}  // namespace msckf

#endif  // INCLUDE_MSCKF_TRANSFORM_MULTIPLY_JACOBIAN_HPP_
