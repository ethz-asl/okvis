#ifndef INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
#define INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
#include <Eigen/Geometry>

#include <msckf/ProjParamOptModels.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/kinematics/operators.hpp>

namespace okvis {

/**
 * @brief obsDirectionJacobian compute the Jacobian of the obsDirection
 *     relative to the camera parameters and its covariance
 * @param obsDirection [x/z, y/z, 1] backprojected undistorted coordinates.
 * @param cameraGeometry
 * @param pixelNoiseStd
 * @param dfj_dXcam
 * @param cov_fj
 * @return false if backprojected direction failed to project onto image.
 */
inline bool obsDirectionJacobian(
    const Eigen::Vector3d& obsDirection,
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry,
    int projOptModelId, double pixelNoiseStd,
    Eigen::Matrix<double, 3, Eigen::Dynamic>* dfj_dXcam,
    Eigen::Matrix3d* cov_fj) {
  const Eigen::Vector3d& fj = obsDirection;
  Eigen::Vector2d imagePoint;
  Eigen::Matrix<double, 2, 3> pointJacobian;
  Eigen::Matrix2Xd intrinsicsJacobian;
  okvis::cameras::CameraBase::ProjectionStatus projectOk =
      cameraGeometry->project(fj, &imagePoint, &pointJacobian, &intrinsicsJacobian);
  if (projectOk != okvis::cameras::CameraBase::ProjectionStatus::Successful)
    return false;
  ProjectionOptKneadIntrinsicJacobian(projOptModelId, &intrinsicsJacobian);
  Eigen::Matrix2d dz_df12 = pointJacobian.topLeftCorner<2, 2>();
  Eigen::Matrix2d df12_dz = dz_df12.inverse();
  int cols = intrinsicsJacobian.cols();
  dfj_dXcam->resize(3, cols);
  dfj_dXcam->topLeftCorner(2, cols) = -df12_dz * intrinsicsJacobian;
  dfj_dXcam->row(2).setZero();
  cov_fj->setZero();
  cov_fj->topLeftCorner<2, 2>() = df12_dz * Eigen::Matrix2d::Identity() *
                                  df12_dz.transpose() * pixelNoiseStd *
                                  pixelNoiseStd;
  return true;
}

class EpipolarJacobian {
 public:
  inline EpipolarJacobian(const Eigen::Matrix3d& R_CjCk,
                          const Eigen::Vector3d& t_CjCk,
                          const Eigen::Vector3d& fj, const Eigen::Vector3d& fk);
  inline double evaluate() const;
  inline void de_dtheta_CjCk(Eigen::Matrix<double, 1, 3>* jac) const;
  inline void de_dfj(Eigen::Matrix<double, 1, 3>* jac) const;
  inline void de_dt_CjCk(Eigen::Matrix<double, 1, 3>* jac) const;
  inline void de_dfk(Eigen::Matrix<double, 1, 3>* jac) const;

 private:
  const Eigen::Matrix3d R_CjCk_;
  const Eigen::Vector3d t_CjCk_;
  const Eigen::Vector3d fj_;  // z = 1
  const Eigen::Vector3d fk_;  // z = 1
};

inline EpipolarJacobian::EpipolarJacobian(const Eigen::Matrix3d& R_CjCk,
                                          const Eigen::Vector3d& t_CjCk,
                                          const Eigen::Vector3d& fj,
                                          const Eigen::Vector3d& fk)
    : R_CjCk_(R_CjCk), t_CjCk_(t_CjCk), fj_(fj), fk_(fk) {}

inline double EpipolarJacobian::evaluate() const {
  // variants of the below expression does not change the Jacobians
  return (R_CjCk_ * fk_).dot(t_CjCk_.cross(fj_));
}

inline void EpipolarJacobian::de_dtheta_CjCk(
    Eigen::Matrix<double, 1, 3>* jac) const {
  *jac = (R_CjCk_ * fk_).transpose() *
         okvis::kinematics::crossMx(t_CjCk_.cross(fj_));
}

inline void EpipolarJacobian::de_dfj(Eigen::Matrix<double, 1, 3>* jac) const {
  *jac = (R_CjCk_ * fk_).transpose() * okvis::kinematics::crossMx(t_CjCk_);
}

inline void EpipolarJacobian::de_dt_CjCk(
    Eigen::Matrix<double, 1, 3>* jac) const {
  *jac = -(R_CjCk_ * fk_).transpose() * okvis::kinematics::crossMx(fj_);
}

inline void EpipolarJacobian::de_dfk(Eigen::Matrix<double, 1, 3>* jac) const {
  *jac = (t_CjCk_.cross(fj_)).transpose() * R_CjCk_;
}
}  // namespace okvis
#endif  // INCLUDE_MSCKF_EPIPOLAR_JACOBIAN_HPP_
