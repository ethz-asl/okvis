
/**
 * @file implementation/EpipolarFactor.hpp
 * @brief Header implementation file for the EpipolarFactor class.
 * @author Jianzhu Huai
 */

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <msckf/EpipolarJacobian.hpp>
#include <msckf/ExtrinsicModels.hpp>
#include <msckf/Measurements.hpp>
#include <msckf/RelativeMotionJacobian.hpp>
#include <msckf/SimpleImuOdometry.hpp>
#include <msckf/JacobianHelpers.hpp>

#include <okvis/ceres/PoseLocalParameterization.hpp>

namespace okvis {
namespace ceres {

template <class GEOMETRY_TYPE, class EXTRINSIC_MODEL,
          class PROJ_INTRINSIC_MODEL>
EpipolarFactor<GEOMETRY_TYPE, EXTRINSIC_MODEL,
               PROJ_INTRINSIC_MODEL>::EpipolarFactor()
    : gravityMag_(9.80665) {}

template <class GEOMETRY_TYPE, class EXTRINSIC_MODEL,
          class PROJ_INTRINSIC_MODEL>
EpipolarFactor<GEOMETRY_TYPE, EXTRINSIC_MODEL, PROJ_INTRINSIC_MODEL>::
    EpipolarFactor(
        std::shared_ptr<const camera_geometry_t> cameraGeometry,
        uint64_t landmarkId,
        const std::vector<Eigen::Vector2d,
                          Eigen::aligned_allocator<Eigen::Vector2d>>&
            measurement12,
        const std::vector<Eigen::Matrix2d,
                          Eigen::aligned_allocator<Eigen::Matrix2d>>&
            covariance12,
        const std::vector<okvis::ImuMeasurementDeque,
                          Eigen::aligned_allocator<okvis::ImuMeasurementDeque>>&
            imuMeasCanopy,
        const okvis::kinematics::Transformation& T_SC_base,
        const std::vector<okvis::Time>& stateEpoch,
        const std::vector<double>& tdAtCreation,
        const std::vector<
            Eigen::Matrix<double, 9, 1>,
            Eigen::aligned_allocator<Eigen::Matrix<double, 9, 1>>>&
            speedAndBiases,
        double gravityMag)
    : measurement_(measurement12),
      covariance_(covariance12),
      imuMeasCanopy_(imuMeasCanopy),
      T_SC_base_(T_SC_base),
      stateEpoch_(stateEpoch),
      tdAtCreation_(tdAtCreation),
      speedAndBiases_(speedAndBiases),
      gravityMag_(gravityMag) {
  ReprojectionErrorBase::setLandmarkId(landmarkId);
  setCameraGeometry(cameraGeometry);
  int imageHeight = cameraGeometryBase_->imageHeight();
  for (int j = 0; j < 2; ++j) {
    double kpN = measurement_[j][1] / imageHeight - 0.5;
    dtij_dtr_[j] = kpN;
  }
}

template <class GEOMETRY_TYPE, class EXTRINSIC_MODEL,
          class PROJ_INTRINSIC_MODEL>
bool EpipolarFactor<GEOMETRY_TYPE, EXTRINSIC_MODEL, PROJ_INTRINSIC_MODEL>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE, class EXTRINSIC_MODEL,
          class PROJ_INTRINSIC_MODEL>
void EpipolarFactor<GEOMETRY_TYPE, EXTRINSIC_MODEL, PROJ_INTRINSIC_MODEL>::
    computePoseAndVelocityAtExposure(
        std::pair<Eigen::Quaternion<double>, Eigen::Matrix<double, 3, 1>>*
            pair_T_WS,
        Eigen::Matrix<double, 6, 1>* velAndOmega,
        double const* const* parameters, int index) const {
  double trLatestEstimate = parameters[5][0];
  double tdLatestEstimate = parameters[6][0];
  Eigen::Matrix<double, 9, 1> speedBgBa = speedAndBiases_[index];

  double relativeFeatureTime =
      tdLatestEstimate + trLatestEstimate * dtij_dtr_[index] - tdAtCreation_[index];

  okvis::Time t_start = stateEpoch_[index];
  okvis::Time t_end = stateEpoch_[index] + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  if (relativeFeatureTime >= wedge) {
    okvis::ceres::predictStates(imuMeasCanopy_[index], gravityMag_, *pair_T_WS,
                                speedBgBa, t_start, t_end);
  } else if (relativeFeatureTime <= -wedge) {
    okvis::ceres::predictStatesBackward(imuMeasCanopy_[index], gravityMag_,
                                        *pair_T_WS, speedBgBa, t_start, t_end);
  }
  velAndOmega->head<3>() = speedBgBa.head<3>();
  okvis::ImuMeasurement queryValue;
  okvis::ceres::interpolateInertialData(imuMeasCanopy_[index], t_end, queryValue);
  queryValue.measurement.gyroscopes -= speedBgBa.segment<3>(3);
  velAndOmega->tail<3>() = queryValue.measurement.gyroscopes;
}

template <class GEOMETRY_TYPE, class EXTRINSIC_MODEL,
          class PROJ_INTRINSIC_MODEL>
bool EpipolarFactor<GEOMETRY_TYPE, EXTRINSIC_MODEL, PROJ_INTRINSIC_MODEL>::
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

  // TODO(jhuai): use extrinsic_model and if needed T_SC_base_
  Eigen::Map<const Eigen::Vector3d> t_SC_S(parameters[2]);
  const Eigen::Quaterniond q_SC(parameters[2][6], parameters[2][3],
                                parameters[2][4], parameters[2][5]);
  // TODO(jhuai): use GEOMETRY_TYPE::NumIntrinsics will lead to undefined
  // reference to NumIntrinsics of 4 instantiated PinholeCamera template class.
  Eigen::VectorXd intrinsics(4 + kDistortionDim);
  if (PROJ_INTRINSIC_MODEL::kNumParams) {
    Eigen::Map<const Eigen::Matrix<double, PROJ_INTRINSIC_MODEL::kNumParams, 1>>
        projIntrinsics(parameters[3]);
    PROJ_INTRINSIC_MODEL::localToGlobal(projIntrinsics, &intrinsics);
  }
  Eigen::Map<const Eigen::Matrix<double, kDistortionDim, 1>>
      distortionIntrinsics(parameters[4]);
  intrinsics.tail<kDistortionDim>() = distortionIntrinsics;
  cameraGeometryBase_->setIntrinsics(intrinsics);

  double trLatestEstimate = parameters[5][0];
  double tdLatestEstimate = parameters[6][0];
  int index = 0;
  std::pair<Eigen::Quaternion<double>, Eigen::Matrix<double, 3, 1>> pair_T_WS1(
      q_WS1, t_WS1_W);
  Eigen::Matrix<double, 6, 1> velAndOmega[2];
  computePoseAndVelocityAtExposure(&pair_T_WS1, &velAndOmega[index], parameters,
                                   index);
  index = 1;
  std::pair<Eigen::Quaternion<double>, Eigen::Matrix<double, 3, 1>> pair_T_WS2(
      q_WS2, t_WS2_W);
  computePoseAndVelocityAtExposure(&pair_T_WS2, &velAndOmega[index], parameters,
                                   index);

  // backProject to compute the obsDirections for the two observations
  bool backProjectOk = true;
  Eigen::Vector3d xy1[2];
  for (int j = 0; j < 2; ++j) {
    bool validDirection =
        cameraGeometryBase_->backProject(measurement_[j], &xy1[j]);
    if (!validDirection) {
      backProjectOk = false;
    }
  }

  // compute Jacobians and covariance for the obs direction
  std::vector<
      Eigen::Matrix<double, 3, Eigen::Dynamic>,
      Eigen::aligned_allocator<Eigen::Matrix<double, 3, Eigen::Dynamic>>>
      dfj_dXcam(2);
  std::vector<Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d>>
      cov_fj(2);
  int projOptModelId = PROJ_INTRINSIC_MODEL::kModelId;
  bool directionJacOk = true;
  for (int j = 0; j < 2; ++j) {
    double pixelNoiseStd = covariance_[j](0, 0);
    bool projectOk =
        obsDirectionJacobian(xy1[j], cameraGeometryBase_, projOptModelId,
                             pixelNoiseStd, &dfj_dXcam[j], &cov_fj[j]);
    if (!projectOk) {
      directionJacOk = false;
    }
  }

  // compute epipolar error with Jacobian and compute the measurement covariance
  std::pair<Eigen::Matrix3d, Eigen::Vector3d> dual_T_SC =
      std::make_pair<Eigen::Matrix3d, Eigen::Vector3d>(
        q_SC.toRotationMatrix(), t_SC_S);
  std::pair<Eigen::Matrix3d, Eigen::Vector3d> dual_T_WS[2] = {
      std::make_pair<Eigen::Matrix3d, Eigen::Vector3d>(q_WS1.toRotationMatrix(), t_WS1_W),
      std::make_pair<Eigen::Matrix3d, Eigen::Vector3d>(q_WS2.toRotationMatrix(), t_WS2_W)};
  RelativeMotionJacobian rmj(dual_T_SC, dual_T_WS[0], dual_T_WS[1]);
  std::pair<Eigen::Matrix3d, Eigen::Vector3d> dual_T_C1C2 = rmj.relativeMotion();
  EpipolarJacobian epj(dual_T_C1C2.first, dual_T_C1C2.second, xy1[0], xy1[1]);
  Eigen::Matrix<double, 1, 3> de_dfj[2];
  epj.de_dfj(&de_dfj[0]);
  epj.de_dfk(&de_dfj[1]);
  const double headObsCovModifier = 4;
  // TODO(jhuai): set and fix squareRootInfo in the constructor
  // TODO(jhuai): account for the IMU noise
  Eigen::Matrix<double, 1, 1> cov_e = de_dfj[0] * cov_fj[0] * de_dfj[0].transpose() * headObsCovModifier +
                 de_dfj[1] * cov_fj[1] * de_dfj[1].transpose();
  bool covOk = cov_e[0] > 1e-8;
  squareRootInformation_ = std::sqrt(1.0 / cov_e[0]);

  bool valid = backProjectOk && directionJacOk && covOk;
  // implicit observation is 0
  residuals[0] = valid ? squareRootInformation_ * epj.evaluate() : 0.0;

  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }

    // compute Jacobians for motion parameters
    Eigen::Matrix<double, 1, 3> de_dtheta_Ctij_Ctik, de_dt_Ctij_Ctik;
    epj.de_dtheta_CjCk(&de_dtheta_Ctij_Ctik);
    epj.de_dt_CjCk(&de_dt_Ctij_Ctik);

    Eigen::Matrix<double, 3, 3> dtheta_dtheta_BC;
    Eigen::Matrix<double, 3, 3> dp_dtheta_BC;
    Eigen::Matrix<double, 3, 3> dp_dt_BC;
    Eigen::Matrix<double, 3, 3> dp_dt_CB;

    Eigen::Matrix<double, 1, Eigen::Dynamic> de_dExtrinsic;
    switch (EXTRINSIC_MODEL::kModelId) {
      case Extrinsic_p_SC_q_SC::kModelId:
        rmj.dtheta_dtheta_BC(&dtheta_dtheta_BC);
        rmj.dp_dtheta_BC(&dp_dtheta_BC);
        rmj.dp_dt_BC(&dp_dt_BC);
        de_dExtrinsic.resize(1, 6);
        de_dExtrinsic.head<3>() = de_dt_Ctij_Ctik * dp_dt_BC;
        de_dExtrinsic.tail<3>() = de_dt_Ctij_Ctik * dp_dtheta_BC +
                                  de_dtheta_Ctij_Ctik * dtheta_dtheta_BC;
        break;
      case Extrinsic_p_CS::kModelId:
        rmj.dp_dt_CB(&dp_dt_CB);
        de_dExtrinsic = de_dt_Ctij_Ctik * dp_dt_CB;
        break;
      case ExtrinsicFixed::kModelId:
      default:
        break;
    }
    Eigen::Matrix<double, 1, Eigen::Dynamic> de_dxcam =
        de_dfj[0] * dfj_dXcam[0] + de_dfj[1] * dfj_dXcam[1];

    // compute Jacobians for time parameters
    Eigen::Matrix<double, 3, 3> dtheta_dtheta_GBtij[2];
    Eigen::Matrix<double, 3, 3> dp_dt_GBtij[2];
    Eigen::Matrix<double, 3, 3> dp_dtheta_GBtij[2];

    rmj.dtheta_dtheta_GBj(&dtheta_dtheta_GBtij[0]);
    rmj.dtheta_dtheta_GBk(&dtheta_dtheta_GBtij[1]);

    rmj.dp_dtheta_GBj(&dp_dtheta_GBtij[0]);
    rmj.dp_dtheta_GBk(&dp_dtheta_GBtij[1]);

    rmj.dp_dt_GBj(&dp_dt_GBtij[0]);
    rmj.dp_dt_GBk(&dp_dt_GBtij[1]);

    Eigen::Matrix<double, 3, 1> dtheta_GBtij_dtij[2];
    Eigen::Matrix<double, 3, 1> dt_GBtij_dtij[2];
    for (int j = 0; j < 2; ++j) {
      dtheta_GBtij_dtij[j] = dual_T_WS[j].first * velAndOmega[j].tail<3>();
      dt_GBtij_dtij[j] = velAndOmega[j].head<3>();
    }

    double de_dtj[2];
    double featureDelay[2];
    for (int j = 0; j < 2; ++j) {
      Eigen::Matrix<double, 1, 1> de_dtj_eigen =
          de_dtheta_Ctij_Ctik *
              (dtheta_dtheta_GBtij[j] * dtheta_GBtij_dtij[j]) +
          de_dt_Ctij_Ctik * (dp_dt_GBtij[j] * dt_GBtij_dtij[j] +
                             dp_dtheta_GBtij[j] * dtheta_GBtij_dtij[j]);
      de_dtj[j] = de_dtj_eigen[0];

      double featureTime =
          tdLatestEstimate + trLatestEstimate * dtij_dtr_[j] - tdAtCreation_[j];
      featureDelay[j] = featureTime;
    }
    double de_dtd = de_dtj[0] + de_dtj[1];
    double de_dtr = de_dtj[0] * dtij_dtr_[0] + de_dtj[1] * dtij_dtr_[1];

    // Jacobians for motion
    Eigen::Matrix<double, 1, 3> de_dp_GBtj[2];
    Eigen::Matrix<double, 1, 3> de_dtheta_GBtj[2];
    Eigen::Matrix<double, 1, 3> de_dv_GBtj[2];

    Eigen::Matrix3d dtheta_GBtij_dtheta_GBtj[2];
    Eigen::Matrix<double, 3, 3> dt_GBtij_dt_GBtj[2];
    Eigen::Matrix3d dt_GBtij_dv_GBtj[2];
    for (int j = 0; j < 2; ++j) {
      dtheta_GBtij_dtheta_GBtj[j].setIdentity();
      dt_GBtij_dt_GBtj[j].setIdentity();
      dt_GBtij_dv_GBtj[j] = Eigen::Matrix3d::Identity() * featureDelay[j];
    }

    for (int j = 0; j < 2; ++j) {
      de_dp_GBtj[j] = de_dt_Ctij_Ctik * dp_dt_GBtij[j] * dt_GBtij_dt_GBtj[j];
      de_dtheta_GBtj[j] = (de_dtheta_Ctij_Ctik * dtheta_dtheta_GBtij[j] +
                           de_dt_Ctij_Ctik * dp_dtheta_GBtij[j]) *
                          dtheta_GBtij_dtheta_GBtj[j];
      de_dv_GBtj[j] = de_dt_Ctij_Ctik * dp_dt_GBtij[j] * dt_GBtij_dv_GBtj[j];
    }

    // assemble the Jacobians, note weight Jacobians with the measurement
    // T_GBtj T_GBtk
    for (int index = 0; index < 2; ++index) {
      if (jacobians[index]) {
        // pseudo inverse of the local parametrization Jacobian:
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
        PoseLocalParameterization::liftJacobian(parameters[index],
                                                J_lift.data());
        Eigen::Matrix<double, 1, 6, Eigen::RowMajor> de_dT_GBtj;
        de_dT_GBtj.head<3>() = de_dp_GBtj[index];
        de_dT_GBtj.tail<3>() = de_dtheta_GBtj[index];
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>>
            jacMapped(jacobians[index]);
        jacMapped = squareRootInformation_ * de_dT_GBtj * J_lift;
        if (jacobiansMinimal && jacobiansMinimal[index]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>>
              jacMinimalMapped(jacobiansMinimal[index]);
          jacMinimalMapped = squareRootInformation_ * de_dT_GBtj;
        }
      }
    }

    // T_SC
    if (jacobians[2]) {
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
      PoseLocalParameterization::liftJacobian(parameters[2], J_lift.data());
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>>
          jacMapped(jacobians[2]);
      jacMapped = squareRootInformation_ * de_dExtrinsic * J_lift;
      if (jacobiansMinimal && jacobiansMinimal[2]) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>>
            jacMinimalMapped(jacobiansMinimal[2]);
        jacMinimalMapped = squareRootInformation_ * de_dExtrinsic;
      }
    }
    // proj intrinsics
    if (jacobians[3]) {
      Eigen::Map<
          Eigen::Matrix<double, kNumResiduals, PROJ_INTRINSIC_MODEL::kNumParams,
                        Eigen::RowMajor>>
          jacMapped(jacobians[3]);
      jacMapped = squareRootInformation_ *
                  de_dxcam.head<PROJ_INTRINSIC_MODEL::kNumParams>();
      if (jacobiansMinimal && jacobiansMinimal[3]) {
        Eigen::Map<
            Eigen::Matrix<double, kNumResiduals,
                          PROJ_INTRINSIC_MODEL::kNumParams, Eigen::RowMajor>>
            jacMinimalMapped(jacobiansMinimal[3]);
        jacMinimalMapped = squareRootInformation_ *
                           de_dxcam.head<PROJ_INTRINSIC_MODEL::kNumParams>();
      }
    }
    // distortion
    if (jacobians[4]) {
      Eigen::Map<
          Eigen::Matrix<double, kNumResiduals, kDistortionDim, Eigen::RowMajor>>
          jacMapped(jacobians[4]);
      jacMapped = squareRootInformation_ * de_dxcam.tail<kDistortionDim>();
      if (jacobiansMinimal && jacobiansMinimal[4]) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, kDistortionDim,
                                 Eigen::RowMajor>>
            jacMinimalMapped(jacobiansMinimal[4]);
        jacMinimalMapped =
            squareRootInformation_ * de_dxcam.tail<kDistortionDim>();
      }
    }
    // tr
    if (jacobians[5]) {
      jacobians[5][0] = squareRootInformation_ * de_dtr;
      if (jacobiansMinimal && jacobiansMinimal[5]) {
        jacobiansMinimal[5][0] = squareRootInformation_ * de_dtr;
      }
    }
    // td
    if (jacobians[6]) {
      jacobians[6][0] = squareRootInformation_ * de_dtd;
      if (jacobiansMinimal && jacobiansMinimal[6]) {
        jacobiansMinimal[6][0] = squareRootInformation_ * de_dtd;
      }
    }
  }
  return true;
}

template <class GEOMETRY_TYPE, class EXTRINSIC_MODEL,
          class PROJ_INTRINSIC_MODEL>
void EpipolarFactor<GEOMETRY_TYPE, EXTRINSIC_MODEL, PROJ_INTRINSIC_MODEL>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 1>(0, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, 1>(1, jacobians, jacobiansMinimal);
  zeroJacobian<EXTRINSIC_MODEL::kGlobalDim, EXTRINSIC_MODEL::kNumParams, 1>(2, jacobians, jacobiansMinimal);
  zeroJacobian<PROJ_INTRINSIC_MODEL::kNumParams,
               PROJ_INTRINSIC_MODEL::kNumParams, 1>(3, jacobians, jacobiansMinimal);
  zeroJacobian<kDistortionDim, kDistortionDim, 1>(4, jacobians, jacobiansMinimal);
  zeroJacobianOne<1>(5, jacobians, jacobiansMinimal);
  zeroJacobianOne<1>(6, jacobians, jacobiansMinimal);
}

}  // namespace ceres
}  // namespace okvis
