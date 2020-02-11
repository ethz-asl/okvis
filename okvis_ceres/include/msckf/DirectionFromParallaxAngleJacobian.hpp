#ifndef INCLUDE_MSCKF_DIRECTION_FROM_PARALLAX_ANGLE_JACOBIAN_HPP_
#define INCLUDE_MSCKF_DIRECTION_FROM_PARALLAX_ANGLE_JACOBIAN_HPP_

#include <msckf/JacobianHelpers.hpp>
#include <msckf/ParallaxAnglePoint.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace msckf {
class DirectionFromParallaxAngleJacobian {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DirectionFromParallaxAngleJacobian() {}

  void initialize(const std::pair<Eigen::Vector3d, Eigen::Quaterniond>& T_WCmi,
                  const Eigen::Vector3d& p_WCai, const Eigen::Vector3d& p_WCtij,
                  const LWF::ParallaxAnglePoint& pap) {
    T_WCmi_ = T_WCmi;
    p_WCai_ = p_WCai;
    p_WCtij_ = p_WCtij;
    pap_ = pap;
    computeIntermediates();
  }

  void initialize(const okvis::kinematics::Transformation& T_WCmi,
                  const Eigen::Vector3d& p_WCai, const Eigen::Vector3d& p_WCtij,
                  const LWF::ParallaxAnglePoint& pap) {
    T_WCmi_.first = T_WCmi.r();
    T_WCmi_.second = T_WCmi.q();
    p_WCai_ = p_WCai;
    p_WCtij_ = p_WCtij;
    pap_ = pap;
    computeIntermediates();
  }

  // The computation of this class assumes that the associate anchor and the
  // observing frame is different. When they are the same, dN_dp_WCai and
  // dN_dp_WCtij should be combined for the Jacobian w.r.t p_WCai or p_WCtij.
  DirectionFromParallaxAngleJacobian(
      const std::pair<Eigen::Vector3d, Eigen::Quaterniond>& T_WCmi,
      const Eigen::Vector3d& p_WCai, const Eigen::Vector3d& p_WCtij,
      const LWF::ParallaxAnglePoint& pap)
      : T_WCmi_(T_WCmi), p_WCai_(p_WCai),
        p_WCtij_(p_WCtij), pap_(pap) {
    computeIntermediates();
  }

  DirectionFromParallaxAngleJacobian(
      const okvis::kinematics::Transformation& T_WCmi,
      const Eigen::Vector3d& p_WCai, const Eigen::Vector3d& p_WCtij,
      const LWF::ParallaxAnglePoint& pap)
      : T_WCmi_(T_WCmi.r(), T_WCmi.q()), p_WCai_(p_WCai),
        p_WCtij_(p_WCtij), pap_(pap) {
    computeIntermediates();
  }

  Eigen::Vector3d evaluate() const {
    return ct_ * axn_norm_ * W_ni_ +
           st_ * (b_ - a_.dot(W_ni_) * W_ni_);
  }

  void dN_dp_WCmi(Eigen::Matrix3d* j) const {
    Eigen::Matrix3d N_b;
    dN_db(&N_b);
    dN_da(j);
    (*j) += N_b;
  }

  void dN_dtheta_WCmi(Eigen::Matrix3d* j) const {
    dN_dWni(j);
    (*j) *= (-okvis::kinematics::crossMx(T_WCmi_.second * pap_.getVec()));
  }

  void dN_dp_WCai(Eigen::Matrix3d* j) const {
    dN_da(j);
    (*j) = -(*j);
  }

  void dN_dp_WCtij(Eigen::Matrix3d* j) const {
    dN_db(j);
    (*j) = -(*j);
  }

  void dN_dni(Eigen::Matrix<double, 3, 2>* j) const {
    Eigen::Matrix3d N_Wni;
    dN_dWni(&N_Wni);
    *j = N_Wni * pap_.n_.getM();
  }

  void dN_dthetai(Eigen::Vector3d* j) const {
    *j = - st_ * axn_norm_ * W_ni_ +
               ct_ * (b_ - a_.dot(W_ni_) * W_ni_);
  }

  void dN_da(Eigen::Matrix3d* j) const {
    *j = -ct_ * W_ni_ * axn_normalized_.transpose() *
             okvis::kinematics::crossMx(W_ni_) -
         st_ * W_ni_ * W_ni_.transpose();
  }

  void dN_db(Eigen::Matrix3d* j) const {
    *j = st_ * Eigen::Matrix3d::Identity();
  }

  void dN_dWni(Eigen::Matrix3d* j) const {
    Eigen::Matrix3d eye = Eigen::Matrix3d::Identity();
    *j = ct_ * (axn_norm_ * eye + W_ni_ * axn_normalized_.transpose() *
               okvis::kinematics::crossMx(a_)) -
        st_ * ((a_.transpose() * W_ni_) * eye + W_ni_ * a_.transpose());
  }

  void computeIntermediates() {
    a_ = T_WCmi_.first - p_WCai_;
    b_ = T_WCmi_.first - p_WCtij_;
    W_ni_ = T_WCmi_.second * pap_.getVec();
    ct_ = pap_.cosTheta();
    st_ = pap_.sinTheta();
    axn_ = a_.cross(W_ni_);
    axn_norm_ = axn_.norm();
    if (axn_norm_ < 1e-6) {
      axn_normalized_ = LWF::NormalVectorElement(W_ni_).getPerp1();
    } else {
      axn_normalized_ = axn_ / axn_norm_;
    }
  }

  std::pair<Eigen::Vector3d, Eigen::Quaterniond> T_WCmi_;
  Eigen::Vector3d p_WCai_;
  Eigen::Vector3d p_WCtij_;
  LWF::ParallaxAnglePoint pap_;

  Eigen::Vector3d a_;
  Eigen::Vector3d b_;
  Eigen::Vector3d W_ni_;
  Eigen::Vector3d axn_;
  double axn_norm_;
  Eigen::Vector3d axn_normalized_;
  double ct_;
  double st_;
};
}  // namespace msckf

#endif  // INCLUDE_MSCKF_DIRECTION_FROM_PARALLAX_ANGLE_JACOBIAN_HPP_
