#ifndef INCLUDE_MSCKF_RELATIVE_MOTION_JACOBIAN_HPP_
#define INCLUDE_MSCKF_RELATIVE_MOTION_JACOBIAN_HPP_

#include <okvis/kinematics/operators.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
/**
 * @brief class to compute Jacobians of
 * \f$ T_{C_{j}C_{k}} = (T_{GB{j}} T_{BC})^{-1} T_{GB{k}} T_{BC} \f$
 * The error states for \f$T_{C_{j}C_{k}}, T_{GB{j}}, T_{GB{k}}, T_{BC}\f$ are
 * defined according to okvis::kinematics::oplus and minus.
 * First order approximation is used in computing the Jacobians. *
*/
class RelativeMotionJacobian {
 public:
  inline RelativeMotionJacobian(const std::pair<Eigen::Matrix3d, Eigen::Vector3d>& T_BC,
                                const std::pair<Eigen::Matrix3d, Eigen::Vector3d>& T_GBj,
                                const std::pair<Eigen::Matrix3d, Eigen::Vector3d>& T_GBk);
  inline RelativeMotionJacobian(const okvis::kinematics::Transformation& T_BC,
                                const okvis::kinematics::Transformation& T_GBj,
                                const okvis::kinematics::Transformation& T_GBk);
  inline std::pair<Eigen::Matrix3d, Eigen::Vector3d> relativeMotion() const;
  inline okvis::kinematics::Transformation relativeMotionT() const;
  inline void dtheta_dtheta_BC(Eigen::Matrix3d* jac) const;
  inline void dtheta_dtheta_GBj(Eigen::Matrix3d* jac) const;
  inline void dtheta_dtheta_GBk(Eigen::Matrix3d* jac) const;
  inline void dp_dtheta_BC(Eigen::Matrix3d* jac) const;
  inline void dp_dtheta_GBj(Eigen::Matrix3d* jac) const;
  inline void dp_dtheta_GBk(Eigen::Matrix3d* jac) const;
  inline void dp_dt_BC(Eigen::Matrix3d* jac) const;
  inline void dp_dt_GBj(Eigen::Matrix3d* jac) const;
  inline void dp_dt_GBk(Eigen::Matrix3d* jac) const;
  inline void dp_dt_CB(Eigen::Matrix3d* jac) const;

 private:
  inline std::pair<Eigen::Matrix3d, Eigen::Vector3d> evaluate() const;
  const std::pair<Eigen::Matrix3d, Eigen::Vector3d> T_BC_;
  const std::pair<Eigen::Matrix3d, Eigen::Vector3d> T_GBj_;
  const std::pair<Eigen::Matrix3d, Eigen::Vector3d> T_GBk_;
  std::pair<Eigen::Matrix3d, Eigen::Vector3d> T_CjCk_;
};

RelativeMotionJacobian::RelativeMotionJacobian(
    const std::pair<Eigen::Matrix3d, Eigen::Vector3d>& T_BC,
    const std::pair<Eigen::Matrix3d, Eigen::Vector3d>& T_GBj,
    const std::pair<Eigen::Matrix3d, Eigen::Vector3d>& T_GBk)
    : T_BC_(T_BC), T_GBj_(T_GBj), T_GBk_(T_GBk) {
  T_CjCk_ = evaluate();
}

RelativeMotionJacobian::RelativeMotionJacobian(
    const okvis::kinematics::Transformation& T_BC,
    const okvis::kinematics::Transformation& T_GBj,
    const okvis::kinematics::Transformation& T_GBk)
    : T_BC_(std::make_pair<Eigen::Matrix3d, Eigen::Vector3d>(
          Eigen::Matrix3d(T_BC.C()), T_BC.r())),
      T_GBj_(std::make_pair<Eigen::Matrix3d, Eigen::Vector3d>(
          Eigen::Matrix3d(T_GBj.C()), T_GBj.r())),
      T_GBk_(std::make_pair<Eigen::Matrix3d, Eigen::Vector3d>(
          Eigen::Matrix3d(T_GBk.C()), T_GBk.r())) {
  T_CjCk_ = evaluate();
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> RelativeMotionJacobian::evaluate()
    const {
  Eigen::Matrix3d C_GCj = T_GBj_.first * T_BC_.first;
  Eigen::Vector3d t_GCj = T_GBj_.first * T_BC_.second + T_GBj_.second;

  Eigen::Matrix3d C_CjG = C_GCj.transpose();
  Eigen::Vector3d t_CjG = - C_CjG * t_GCj;

  Eigen::Matrix3d C_GCk = T_GBk_.first * T_BC_.first;
  Eigen::Vector3d t_GCk = T_GBk_.first * T_BC_.second + T_GBk_.second;

  return std::make_pair<Eigen::Matrix3d, Eigen::Vector3d>(
      C_CjG * C_GCk, C_CjG * t_GCk + t_CjG);
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d>
RelativeMotionJacobian::relativeMotion() const {
  return T_CjCk_;
}

okvis::kinematics::Transformation
RelativeMotionJacobian::relativeMotionT() const {
  return okvis::kinematics::Transformation(
      T_CjCk_.second, Eigen::Quaterniond(T_CjCk_.first));
}

void RelativeMotionJacobian::dtheta_dtheta_BC(
    Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_BjBk = T_GBj_.first.transpose() * T_GBk_.first;
  *jac = T_BC_.first.transpose() * (R_BjBk - Eigen::Matrix3d::Identity());
}

void RelativeMotionJacobian::dtheta_dtheta_GBj(
    Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_GCj = T_GBj_.first * T_BC_.first;
  *jac = -R_GCj.transpose();
}

void RelativeMotionJacobian::dtheta_dtheta_GBk(
    Eigen::Matrix3d* jac) const {
  dtheta_dtheta_GBj(jac);
  *jac = -(*jac);
}

void RelativeMotionJacobian::dp_dtheta_BC(Eigen::Matrix3d* jac) const {
  *jac = okvis::kinematics::crossMx(T_CjCk_.second) * T_BC_.first.transpose();
}

void RelativeMotionJacobian::dp_dtheta_GBj(Eigen::Matrix3d* jac) const {
  Eigen::Vector3d t_GCk = T_GBk_.first * T_BC_.second + T_GBk_.second;
  Eigen::Matrix3d R_GCj = T_GBj_.first * T_BC_.first;
  *jac = R_GCj.transpose() * okvis::kinematics::crossMx(t_GCk - T_GBj_.second);
}

void RelativeMotionJacobian::dp_dtheta_GBk(Eigen::Matrix3d* jac) const {
  Eigen::Matrix3d R_GCj = T_GBj_.first * T_BC_.first;
  *jac =
      -R_GCj.transpose() * okvis::kinematics::crossMx(T_GBk_.first * T_BC_.second);
}

void RelativeMotionJacobian::dp_dt_BC(Eigen::Matrix3d* jac) const {
  *jac = (T_CjCk_.first - Eigen::Matrix3d::Identity()) * T_BC_.first.transpose();
}

void RelativeMotionJacobian::dp_dt_GBj(Eigen::Matrix3d* jac) const {
  *jac = -(T_GBj_.first * T_BC_.first).transpose();
}

void RelativeMotionJacobian::dp_dt_GBk(Eigen::Matrix3d* jac) const {
  *jac = (T_GBj_.first * T_BC_.first).transpose();
}

void RelativeMotionJacobian::dp_dt_CB(Eigen::Matrix3d* jac) const {
  *jac = Eigen::Matrix3d::Identity() - T_CjCk_.first;
}
}  // namespace okvis
#endif  // INCLUDE_MSCKF_RELATIVE_MOTION_JACOBIAN_HPP_
