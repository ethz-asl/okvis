#ifndef INCLUDE_MSCKF_SIMPLE_IMU_PROPAGATION_JACOBIAN_HPP_
#define INCLUDE_MSCKF_SIMPLE_IMU_PROPAGATION_JACOBIAN_HPP_

#include <Eigen/Core>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace msckf {
/**
 * @brief SimpleImuPropagationJacobian computes Jacobians of propagated states
 * relative to initial states.
 * Error states are defined by okvis::kinematics::oplus and minus.
 * @warning: this function assumes that endEpoch and startEpoch differ small, e.g.,
 * less than 0.04 sec.
 */
class SimpleImuPropagationJacobian
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SimpleImuPropagationJacobian() {}

  void initialize(const okvis::Time startEpoch,
                  const okvis::Time endEpoch,
                  const okvis::kinematics::Transformation& endT_WB,
                  const Eigen::Matrix<double, 3, 1>& endV_W,
                  const Eigen::Matrix<double, 3, 1>& endOmega_B) {
    startEpoch_ = startEpoch;
    endEpoch_ = endEpoch;
    endp_WB_B_ = endT_WB.r();
    endq_WB_ = endT_WB.q();
    endV_WB_W_ = endV_W;
    endOmega_WB_B_ = endOmega_B;
  }

  SimpleImuPropagationJacobian(const okvis::Time startEpoch,
                               const okvis::Time endEpoch,
                               const okvis::kinematics::Transformation& endT_WB,
                               const Eigen::Matrix<double, 3, 1>& endV_W,
                               const Eigen::Matrix<double, 3, 1>& endOmega_B) :
    startEpoch_(startEpoch), endEpoch_(endEpoch),
    endp_WB_B_(endT_WB.r()),
    endq_WB_(endT_WB.q()),
    endV_WB_W_(endV_W),
    endOmega_WB_B_(endOmega_B)
  {}

  inline void dp_dt_WB(Eigen::Matrix3d* j) {
    j->setIdentity();
  }

  inline void dp_dtheta_WB(Eigen::Matrix3d* j) {
    j->setZero();
  }

  inline void dp_dv_WB(Eigen::Matrix3d* j) {
    j->setIdentity();
    double stride = (endEpoch_ - startEpoch_).toSec();
    (*j)(0, 0) = stride;
    (*j)(1, 1) = stride;
    (*j)(2, 2) = stride;
  }

  inline void dtheta_dt_WB(Eigen::Matrix3d* j) {
    j->setZero();
  }

  inline void dtheta_dtheta_WB(Eigen::Matrix3d* j) {
    j->setIdentity();
  }

  inline void dtheta_dv_WB(Eigen::Matrix3d* j) {
    j->setZero();
  }

  inline void dp_dt(Eigen::Vector3d* j) {
    (*j) = endV_WB_W_;
  }

  inline void dtheta_dt(Eigen::Vector3d* j) {
    (*j) = endq_WB_ * endOmega_WB_B_;
  }

  static Eigen::Vector3d dp_dt(const Eigen::Vector3d& endV_WB_W) {
    return endV_WB_W;
  }

  static Eigen::Vector3d dtheta_dt(const Eigen::Vector3d& endOmega_WB_B,
                                   const Eigen::Quaterniond& endq_WB) {
    return endq_WB * endOmega_WB_B;
  }

  static Eigen::Matrix3d dp_dv_WB(double deltaTime) {
    Eigen::Matrix3d j = Eigen::Matrix3d::Identity();
    j(0, 0) = deltaTime;
    j(1, 1) = deltaTime;
    j(2, 2) = deltaTime;
    return j;
  }

private:
  okvis::Time startEpoch_;
  okvis::Time endEpoch_;
  Eigen::Matrix<double, 3, 1> endp_WB_B_;
  Eigen::Quaterniond endq_WB_;
  Eigen::Matrix<double, 3, 1> endV_WB_W_;
  Eigen::Matrix<double, 3, 1> endOmega_WB_B_;
};
} // namespace msckf

#endif // INCLUDE_MSCKF_SIMPLE_IMU_PROPAGATION_JACOBIAN_HPP_
