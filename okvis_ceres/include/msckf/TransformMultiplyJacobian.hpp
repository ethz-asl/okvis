#ifndef INCLUDE_MSCKF_TRANSFORM_MULTIPLY_JACOBIAN_HPP_
#define INCLUDE_MSCKF_TRANSFORM_MULTIPLY_JACOBIAN_HPP_

#include <okvis/kinematics/Transformation.hpp>

namespace msckf {
// Jacobians for $T_z = T_x * T_y$.
// Oplus and Ominus for $T_x$ $T_y$ and $T_z$ are defined as in Transformation.
// T_z = Oplus(\hat{T}_z, \delta z)
// \delta z = Ominus(T_z, \hat{T}_z)
class TransformMultiplyJacobian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  TransformMultiplyJacobian(const okvis::kinematics::Transformation& T_AB,
                            const okvis::kinematics::Transformation& T_BC)
      : T_AB_(T_AB.r(), T_AB.q()), T_BC_(T_BC.r(), T_BC.q()) {}

  TransformMultiplyJacobian(const okvis::kinematics::Transformation& T_AB,
                            const okvis::kinematics::Transformation& T_BC,
                            const Eigen::Vector3d& v_AB,
                            const Eigen::Vector3d& omega_B)
      : T_AB_(T_AB.r(), T_AB.q()),
        T_BC_(T_BC.r(), T_BC.q()),
        v_AB_(v_AB),
        omega_AB_B_(omega_B) {}

  TransformMultiplyJacobian(const std::pair<Eigen::Vector3d, Eigen::Quaterniond>& T_AB,
                            const std::pair<Eigen::Vector3d, Eigen::Quaterniond>& T_BC)
      : T_AB_(T_AB.first, T_AB.second), T_BC_(T_BC.first, T_BC.second) {}

  TransformMultiplyJacobian() {}

  void initialize(const okvis::kinematics::Transformation& T_AB,
                  const okvis::kinematics::Transformation& T_BC) {
    T_AB_.first = T_AB.r();
    T_AB_.second = T_AB.q();
    T_BC_.first = T_BC.r();
    T_BC_.second = T_BC.q();
  }

  void initialize(const std::pair<Eigen::Vector3d, Eigen::Quaterniond>& T_AB,
                  const std::pair<Eigen::Vector3d, Eigen::Quaterniond>& T_BC) {
    T_AB_.first = T_AB.first;
    T_AB_.second = T_AB.second;
    T_BC_.first = T_BC.first;
    T_BC_.second = T_BC.second;
  }

  void setVelocity(const Eigen::Vector3d& v_AB, const Eigen::Vector3d& omega_B) {
    v_AB_ = v_AB;
    omega_AB_B_ = omega_B;
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

  void dtheta_dp_AB(Eigen::Matrix3d* j) { j->setZero(); }

  void dtheta_dtheta_BC(Eigen::Matrix3d* j) {
    *j = T_AB_.second.toRotationMatrix();
  }

  void dtheta_dp_BC(Eigen::Matrix3d* j) { j->setZero(); }

  void dtheta_dt(Eigen::Vector3d* j) {
    *j = T_AB_.second * omega_AB_B_;
  }

  void dp_dtheta_AB(Eigen::Matrix3d* j) {
    *j = okvis::kinematics::crossMx(T_AB_.second * -T_BC_.first);
  }

  void dp_dp_AB(Eigen::Matrix3d* j) { j->setIdentity(); }

  void dp_dtheta_BC(Eigen::Matrix3d* j) { j->setZero(); }

  void dp_dp_BC(Eigen::Matrix3d* j) { *j = T_AB_.second.toRotationMatrix(); }

  void dp_dt(Eigen::Vector3d* j) {
    *j = v_AB_ + T_AB_.second * omega_AB_B_.cross(T_BC_.first);
  }

 private:
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_AB_;
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_BC_;
  Eigen::Vector3d v_AB_;
  Eigen::Vector3d omega_AB_B_;
  // We assume T_BC_ is constant, thus its velocities are zero.
};
}  // namespace msckf

#endif  // INCLUDE_MSCKF_TRANSFORM_MULTIPLY_JACOBIAN_HPP_
