#ifndef INCLUDE_MSCKF_SIMPLE_IMU_PROPAGATION_JACOBIAN_HPP_
#define INCLUDE_MSCKF_SIMPLE_IMU_PROPAGATION_JACOBIAN_HPP_

#include <Eigen/Core>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace msckf {
// compute Jacobians of propagated states relative to initial states.
// All Jacobians are computed on the tangent space.
// Warn: this function assumes that endEpoch and startEpoch differ small,
// e.g., less than 0.04 sec.
class SimpleImuPropagationJacobian
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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

  static void dp_dt(Eigen::Vector3d endV_WB_W, Eigen::Vector3d* j) {
    (*j) = endV_WB_W;
  }

  static void dtheta_dt(Eigen::Vector3d endOmega_WB_B,
                        Eigen::Quaterniond endq_WB, Eigen::Vector3d* j) {
    (*j) = endq_WB * endOmega_WB_B;
  }

  static void dp_dv_WB(double deltaTime, Eigen::Matrix3d* j) {
    j->setIdentity();
    double stride = deltaTime;
    (*j)(0, 0) = stride;
    (*j)(1, 1) = stride;
    (*j)(2, 2) = stride;
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
