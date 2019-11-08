
/**
 * @file implementation/EpipolarFactor.hpp
 * @brief Header implementation file for the EpipolarFactor class.
 * @author Jianzhu Huai
 */

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <msckf/EpipolarJacobian.hpp>
#include <msckf/Measurements.hpp>
#include <msckf/SimpleImuOdometry.hpp>
#include <msckf/JacobianHelpers.hpp>

namespace okvis {
namespace ceres {

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
EpipolarFactor<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL,
                    EXTRINSIC_MODEL>::EpipolarFactor()
    : gravityMag_(9.80665) {}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
EpipolarFactor<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    EpipolarFactor(std::shared_ptr<const camera_geometry_t> cameraGeometry,
                   const std::vector<Eigen::Vector2d,
                                     Eigen::aligned_allocator<Eigen::Vector2d>>&
                       measurement12,
                   const std::vector<Eigen::Matrix2d,
                                     Eigen::aligned_allocator<Eigen::Matrix2d>>&
                       covariance12,
                   std::vector<std::shared_ptr<const okvis::ImuMeasurementDeque>> imuMeasCanopy,
                   const okvis::kinematics::Transformation& T_SC_base,
                   const std::vector<okvis::Time>& stateEpoch,
                   const std::vector<double>& tdAtCreation, double gravityMag)
    : measurement_(measurement12),
      covariance_(covariance12),
      T_SC_base_(T_SC_base),
      imuMeasCanopy_(imuMeasCanopy),
      stateEpoch_(stateEpoch),
      tdAtCreation_(tdAtCreation),
      gravityMag_(gravityMag) {
  setCameraGeometry(cameraGeometry);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
bool EpipolarFactor<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
void EpipolarFactor<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    computePoseAtExposure(std::pair<Eigen::Quaternion<double>,
                                    Eigen::Matrix<double, 3, 1>>* pairT_WS,
                          double const* const* parameters, int index) const {
  double trLatestEstimate = parameters[5][0];
  double tdLatestEstimate = parameters[6][0];
  Eigen::Matrix<double, 9, 1> speedBgBa =
      Eigen::Map<const Eigen::Matrix<double, 9, 1>>(parameters[7 + index]);
  double ypixel(measurement_[index][1]);
  uint32_t height = cameraGeometryBase_->imageHeight();
  double kpN = ypixel / height - 0.5;
  double relativeFeatureTime =
      tdLatestEstimate + trLatestEstimate * kpN - tdAtCreation_[index];

  okvis::Time t_start = stateEpoch_[index];
  okvis::Time t_end = stateEpoch_[index] + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  if (relativeFeatureTime >= wedge) {
    okvis::ceres::predictStates(*imuMeasCanopy_[index], gravityMag_, *pairT_WS,
                                speedBgBa, t_start, t_end);
  } else if (relativeFeatureTime <= -wedge) {
    okvis::ceres::predictStatesBackward(*imuMeasCanopy_[index], gravityMag_,
                                        *pairT_WS, speedBgBa, t_start, t_end);
  }
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
bool EpipolarFactor<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  // We avoid the use of okvis::kinematics::Transformation here due to
  // quaternion normalization and so forth. This only matters in order to be
  // able to check Jacobians with numeric differentiation chained, first w.r.t.
  // q and then d_alpha.

  // pose: world to sensor transformation
  Eigen::Map<const Eigen::Vector3d> t_WS1_W(parameters[0]);
  const Eigen::Quaterniond q_WS1(parameters[0][6], parameters[0][3],
                                 parameters[0][4], parameters[0][5]);
  Eigen::Map<const Eigen::Vector3d> t_WS2_W(parameters[1]);
  const Eigen::Quaterniond q_WS2(parameters[1][6], parameters[1][3],
                                 parameters[1][4], parameters[1][5]);

  // TODO(jhuai): use extrinsic_model and if needed T_SC_base
  Eigen::Map<const Eigen::Vector3d> t_SC_S(parameters[2]);
  const Eigen::Quaterniond q_SC(parameters[2][6], parameters[2][3],
                                parameters[2][4], parameters[2][5]);

  Eigen::VectorXd intrinsics(GEOMETRY_TYPE::NumIntrinsics);
  if (PROJ_INTRINSIC_MODEL::kNumParams) {
    Eigen::Map<const Eigen::Matrix<double, PROJ_INTRINSIC_MODEL::kNumParams, 1>>
        projIntrinsics(parameters[3]);
    PROJ_INTRINSIC_MODEL::localToGlobal(projIntrinsics, &intrinsics);
  }
  Eigen::Map<const Eigen::Matrix<double, kDistortionDim, 1>>
      distortionIntrinsics(parameters[4]);
  intrinsics.tail<kDistortionDim>() = distortionIntrinsics;
  cameraGeometryBase_->setIntrinsics(intrinsics);

  int index = 0;
  std::pair<Eigen::Quaternion<double>, Eigen::Matrix<double, 3, 1>> pairT_WS1(
      q_WS1, t_WS1_W);
  computePoseAtExposure(&pairT_WS1, parameters, index);
  index = 1;
  std::pair<Eigen::Quaternion<double>, Eigen::Matrix<double, 3, 1>> pairT_WS2(
      q_WS2, t_WS2_W);
  computePoseAtExposure(&pairT_WS2, parameters, index);

  // backProject to compute the obsDirections for the two observations

  // compute Jacobians and covariance for the obs direction
  // okvis::obsDirectionJacobian();

  // compute epipolar error with Jacobian and compute the measurement covariance
  // okvis::EpipolarJacobian epj();
  // residuals[0] = squareRootInformation_ * error;


  // compute the Jacobians of relative pose relative to pose errors,
  // extrinsics and time offset, rolling shutter skew

  // assemble the Jacobians
  // check if the previous steps are valid, use setJacobiansZero if needed

//  if (jacobians != NULL) {
//    if (!valid) {
//      setJacobiansZero(jacobians, jacobiansMinimal);
//      return true;
//    }
//  }

  return true;
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
void EpipolarFactor<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 1>(0, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, 1>(1, jacobians, jacobiansMinimal);
  zeroJacobian<7, EXTRINSIC_MODEL::kNumParams, 1>(2, jacobians, jacobiansMinimal);
  zeroJacobian<PROJ_INTRINSIC_MODEL::kNumParams,
               PROJ_INTRINSIC_MODEL::kNumParams, 1>(3, jacobians, jacobiansMinimal);
  zeroJacobian<kDistortionDim, kDistortionDim, 1>(4, jacobians, jacobiansMinimal);
  zeroJacobianOne<1>(5, jacobians, jacobiansMinimal);
  zeroJacobianOne<1>(6, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 1>(7, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 1>(8, jacobians, jacobiansMinimal);
}

}  // namespace ceres
}  // namespace okvis
