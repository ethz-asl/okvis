
/**
 * @file implementation/RsReprojectionError.hpp
 * @brief Header implementation file for the RsReprojectionError class.
 * @author Jianzhu Huai
 */
#include "ceres/internal/autodiff.h"

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

#include <msckf/JacobianHelpers.hpp>
#include <msckf/Measurements.hpp>
#include <msckf/SimpleImuOdometry.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL,
                    EXTRINSIC_MODEL>::RsReprojectionError()
    : gravityMag_(9.80665) {}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    RsReprojectionError(
        std::shared_ptr<const camera_geometry_t> cameraGeometry,
        uint64_t cameraId, const measurement_t& measurement,
        const covariance_t& information,
        const okvis::ImuMeasurementDeque& imuMeasCanopy,
        const okvis::kinematics::Transformation& T_SC_base,
        okvis::Time stateEpoch, double tdAtCreation, double gravityMag)
    : T_SC_base_(T_SC_base),
      imuMeasCanopy_(imuMeasCanopy),
      stateEpoch_(stateEpoch),
      tdAtCreation_(tdAtCreation),
      gravityMag_(gravityMag) {
  setCameraId(cameraId);
  setMeasurement(measurement);
  setInformation(information);
  setCameraGeometry(cameraGeometry);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
void RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    setInformation(const covariance_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error
  // weighting
  Eigen::LLT<Eigen::Matrix2d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
bool RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
bool RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  // We avoid the use of okvis::kinematics::Transformation here due to
  // quaternion normalization and so forth. This only matters in order to be
  // able to check Jacobians with numeric differentiation chained, first w.r.t.
  // q and then d_alpha.

  Eigen::Map<const Eigen::Vector3d> t_WS_W0(parameters[0]);
  const Eigen::Quaterniond q_WS0(parameters[0][6], parameters[0][3],
                                 parameters[0][4], parameters[0][5]);

  // the point in world coordinates
  Eigen::Map<const Eigen::Vector4d> hp_W(&parameters[1][0]);

  // TODO(jhuai): use Extrinsic_model
  Eigen::Map<const Eigen::Vector3d> t_SC_S(parameters[2]);
  const Eigen::Quaterniond q_SC(parameters[2][6], parameters[2][3],
                                parameters[2][4], parameters[2][5]);
  Eigen::VectorXd intrinsics(GEOMETRY_TYPE::NumIntrinsics);

  Eigen::Map<const Eigen::Matrix<double, PROJ_INTRINSIC_MODEL::kNumParams, 1>>
      projIntrinsics(parameters[3]);
  PROJ_INTRINSIC_MODEL::localToGlobal(projIntrinsics, &intrinsics);

  Eigen::Map<const Eigen::Matrix<double, kDistortionDim, 1>>
      distortionIntrinsics(parameters[4]);
  intrinsics.tail<kDistortionDim>() = distortionIntrinsics;
  cameraGeometryBase_->setIntrinsics(intrinsics);

  double trLatestEstimate = parameters[5][0];
  double tdLatestEstimate = parameters[6][0];

  double ypixel(measurement_[1]);
  uint32_t height = cameraGeometryBase_->imageHeight();
  double kpN = ypixel / height - 0.5;
  double relativeFeatureTime = tdLatestEstimate + trLatestEstimate * kpN - tdAtCreation_;
  std::pair<Eigen::Quaternion<double>, Eigen::Matrix<double, 3, 1>> pairT_WS(
      q_WS0, t_WS_W0);
  Eigen::Matrix<double, 9, 1> speedBgBa =
      Eigen::Map<const Eigen::Matrix<double, 9, 1>>(parameters[7]);

  okvis::Time t_start = stateEpoch_;
  okvis::Time t_end = stateEpoch_ + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  if (relativeFeatureTime >= wedge) {
    okvis::ceres::predictStates(imuMeasCanopy_, gravityMag_, pairT_WS,
                                speedBgBa, t_start, t_end);
  } else if (relativeFeatureTime <= -wedge) {
    okvis::ceres::predictStatesBackward(imuMeasCanopy_, gravityMag_, pairT_WS,
                                        speedBgBa, t_start, t_end);
  }

  Eigen::Quaterniond q_WS = pairT_WS.first;
  Eigen::Vector3d t_WS_W = pairT_WS.second;

  // transform the point into the camera:
  Eigen::Matrix3d C_SC = q_SC.toRotationMatrix();
  Eigen::Matrix3d C_CS = C_SC.transpose();
  Eigen::Matrix4d T_CS = Eigen::Matrix4d::Identity();
  T_CS.topLeftCorner<3, 3>() = C_CS;
  T_CS.topRightCorner<3, 1>() = -C_CS * t_SC_S;
  Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();
  Eigen::Matrix3d C_SW = C_WS.transpose();
  Eigen::Matrix4d T_SW = Eigen::Matrix4d::Identity();
  T_SW.topLeftCorner<3, 3>() = C_SW;
  T_SW.topRightCorner<3, 1>() = -C_SW * t_WS_W;
  Eigen::Vector4d hp_S = T_SW * hp_W;
  Eigen::Vector4d hp_C = T_CS * hp_S;

  // calculate the reprojection error
  measurement_t kp;
  Eigen::Matrix<double, 2, 4> Jh;
  Eigen::Matrix<double, 2, 4> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL) {
    cameraGeometryBase_->projectHomogeneous(hp_C, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  } else {
    cameraGeometryBase_->projectHomogeneous(hp_C, &kp);
  }

  measurement_t error = measurement_ - kp;

  // weight:
  measurement_t weighted_error = squareRootInformation_ * error;

  // assign:
  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  // check validity:
  bool valid = true;
  if (fabs(hp_C[3]) > 1.0e-8) {
    Eigen::Vector3d p_C = hp_C.template head<3>() / hp_C[3];
    if (p_C[2] < 0.2) {  // 20 cm - not very generic... but reasonable
      // std::cout<<"INVALID POINT"<<std::endl;
      valid = false;
    }
  }

  // calculate jacobians, if required
  // This is pretty close to Paul Furgale's thesis. eq. 3.100 on page 40
  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }
    Eigen::Matrix<double, 4, 6> dhC_deltaTWS;
    Eigen::Matrix<double, 4, 4> dhC_deltahpW;
    Eigen::Matrix<double, 4, 6> dhC_deltaTSC;
    Eigen::Vector4d dhC_td;
    Eigen::Matrix<double, 4, 9> dhC_sb;

    Eigen::Vector3d p_W = hp_W.head<3>() - t_WS_W * hp_W[3];
    Eigen::Matrix<double, 4, 6> dhS_deltaTWS;

    dhS_deltaTWS.topLeftCorner<3, 3>() = -C_SW * hp_W[3];
    dhS_deltaTWS.topRightCorner<3, 3>() =
        C_SW * okvis::kinematics::crossMx(p_W);
    dhS_deltaTWS.row(3).setZero();
    dhC_deltaTWS = T_CS * dhS_deltaTWS;

    dhC_deltahpW = T_CS * T_SW;

    Eigen::Vector3d p_S = hp_S.head<3>() - t_SC_S * hp_S[3];
    dhC_deltaTSC.topLeftCorner<3, 3>() = -C_CS * hp_S[3];
    dhC_deltaTSC.topRightCorner<3, 3>() =
        C_CS * okvis::kinematics::crossMx(p_S);
    dhC_deltaTSC.row(3).setZero();

    okvis::ImuMeasurement queryValue;
    okvis::ceres::interpolateInertialData(imuMeasCanopy_, t_end, queryValue);
    queryValue.measurement.gyroscopes -= speedBgBa.segment<3>(3);
    Eigen::Vector3d p =
        okvis::kinematics::crossMx(queryValue.measurement.gyroscopes) *
            hp_S.head<3>() +
        C_SW * speedBgBa.head<3>() * hp_W[3];
    dhC_td.head<3>() = -C_CS * p;
    dhC_td[3] = 0;

    Eigen::Matrix3d dhC_vW = -C_CS * C_SW * relativeFeatureTime * hp_W[3];
    Eigen::Matrix3d dhC_bg =
        -C_CS * C_SW *
        okvis::kinematics::crossMx(hp_W.head<3>() - hp_W[3] * t_WS_W) *
        relativeFeatureTime * q_WS0.toRotationMatrix();

    dhC_sb.row(3).setZero();
    dhC_sb.topRightCorner<3, 3>().setZero();
    dhC_sb.topLeftCorner<3, 3>() = dhC_vW;
    dhC_sb.block<3, 3>(0, 3) = dhC_bg;

    assignJacobians(parameters, jacobians, jacobiansMinimal, Jh_weighted,
                    Jpi_weighted, dhC_deltaTWS, dhC_deltahpW, dhC_deltaTSC,
                    dhC_td, kpN, dhC_sb);
  }
  return true;
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
void RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 2>(0, jacobians, jacobiansMinimal);
  zeroJacobian<4, 3, 2>(1, jacobians, jacobiansMinimal);
  zeroJacobian<EXTRINSIC_MODEL::kGlobalDim, EXTRINSIC_MODEL::kNumParams, 2>(2, jacobians, jacobiansMinimal);
  zeroJacobian<PROJ_INTRINSIC_MODEL::kNumParams,
               PROJ_INTRINSIC_MODEL::kNumParams, 2>(3, jacobians,
                                                    jacobiansMinimal);
  zeroJacobian<kDistortionDim, kDistortionDim, 2>(4, jacobians,
                                                  jacobiansMinimal);
  zeroJacobianOne<2>(5, jacobians, jacobiansMinimal);
  zeroJacobianOne<2>(6, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 2>(7, jacobians, jacobiansMinimal);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
void RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    assignJacobians(
        double const* const* parameters, double** jacobians,
        double** jacobiansMinimal,
        const Eigen::Matrix<double, 2, 4>& Jh_weighted,
        const Eigen::Matrix<double, 2, Eigen::Dynamic>& Jpi_weighted,
        const Eigen::Matrix<double, 4, 6>& dhC_deltaTWS,
        const Eigen::Matrix<double, 4, 4>& dhC_deltahpW,
        const Eigen::Matrix<double, 4, 6>& dhC_deltaTSC,
        const Eigen::Vector4d& dhC_td, double kpN,
        const Eigen::Matrix<double, 4, 9>& dhC_sb) const {
  if (jacobians[0] != NULL) {
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
    J0_minimal = -Jh_weighted * dhC_deltaTWS;
    // pseudo inverse of the local parametrization Jacobian
    Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
    PoseLocalParameterization::liftJacobian(parameters[0], J_lift.data());

    // hallucinate Jacobian w.r.t. state
    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J0(jacobians[0]);
    J0 = J0_minimal * J_lift;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[0] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            J0_minimal_mapped(jacobiansMinimal[0]);
        J0_minimal_mapped = J0_minimal;
      }
    }
  }

  if (jacobians[1] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J1(jacobians[1]);
    J1 = -Jh_weighted * dhC_deltahpW;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[1] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>
            J1_minimal_mapped(jacobiansMinimal[1]);
        Eigen::Matrix<double, 4, 3> S;
        S.setZero();
        S.topLeftCorner<3, 3>().setIdentity();
        J1_minimal_mapped = J1 * S;
      }
    }
  }

  if (jacobians[2] != NULL) {
    // compute the minimal version
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J2_minimal;
    J2_minimal = -Jh_weighted * dhC_deltaTSC;

    // pseudo inverse of the local parametrization Jacobian:
    Eigen::Matrix<double, 6, 7, Eigen::RowMajor> J_lift;
    PoseLocalParameterization::liftJacobian(parameters[2], J_lift.data());

    // hallucinate Jacobian w.r.t. state
    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J2(jacobians[2]);
    J2 = J2_minimal * J_lift;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>>
            J2_minimal_mapped(jacobiansMinimal[2]);
        J2_minimal_mapped = J2_minimal;
      }
    }
  }

  // camera intrinsics
  if (jacobians[3] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, PROJ_INTRINSIC_MODEL::kNumParams,
        Eigen::RowMajor>> J1(jacobians[3]);
    Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted_copy = Jpi_weighted;
    PROJ_INTRINSIC_MODEL::kneadIntrinsicJacobian(&Jpi_weighted_copy);
    J1 = -Jpi_weighted_copy
        .template topLeftCorner<2, PROJ_INTRINSIC_MODEL::kNumParams>();
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[3] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, PROJ_INTRINSIC_MODEL::kNumParams,
            Eigen::RowMajor>> J1_minimal_mapped(jacobiansMinimal[3]);
        J1_minimal_mapped = J1;
      }
    }
  }

  if (jacobians[4] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, kDistortionDim, Eigen::RowMajor>> J1(
        jacobians[4]);
    J1 = -Jpi_weighted.template topRightCorner<2, kDistortionDim>();
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[4] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, kDistortionDim, Eigen::RowMajor>>
            J1_minimal_mapped(jacobiansMinimal[4]);
        J1_minimal_mapped = J1;
      }
    }
  }

  if (jacobians[5] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 1>> J1(jacobians[5]);
    J1 = -Jh_weighted * dhC_td * kpN;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[5] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>>
            J1_minimal_mapped(jacobiansMinimal[5]);
        J1_minimal_mapped = J1;
      }
    }
  }

  // t_d
  if (jacobians[6] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 1>> J1(jacobians[6]);
    J1 = -Jh_weighted * dhC_td;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[6] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 1>> J1_minimal_mapped(
            jacobiansMinimal[6]);
        J1_minimal_mapped = J1;
      }
    }
  }

  // speed and gyro biases and accel biases
  if (jacobians[7] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J1(jacobians[7]);
    J1 = -Jh_weighted * dhC_sb;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[7] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
            J1_minimal_mapped(jacobiansMinimal[7]);
        J1_minimal_mapped = J1;
      }
    }
  }
}


// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation via autodiff
template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
bool RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
    EvaluateWithMinimalJacobiansAutoDiff(double const* const* parameters,
                                         double* residuals, double** jacobians,
                                         double** jacobiansMinimal) const {
  const int numOutputs = 4;
  double deltaTWS[6] = {0};
  double deltaTSC[6] = {0};
  double const* const expandedParams[] = {
      parameters[0], parameters[1], parameters[2],
      parameters[5], parameters[6], parameters[7], deltaTWS, deltaTSC};

  double php_C[numOutputs];
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_deltaTWS_full;
  Eigen::Matrix<double, numOutputs, 4, Eigen::RowMajor> dhC_deltahpW;
  Eigen::Matrix<double, numOutputs, 7, Eigen::RowMajor> dhC_deltaTSC_full;
  Eigen::Matrix<double, numOutputs, PROJ_INTRINSIC_MODEL::kNumParams,
                Eigen::RowMajor>
      dhC_projIntrinsic;
  Eigen::Matrix<double, numOutputs, kDistortionDim, Eigen::RowMajor>
      dhC_distortion;
  Eigen::Matrix<double, numOutputs, 1> dhC_tr;
  Eigen::Matrix<double, numOutputs, 1> dhC_td;
  Eigen::Matrix<double, numOutputs, 9, Eigen::RowMajor> dhC_sb;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_deltaTWS;
  Eigen::Matrix<double, numOutputs, 6, Eigen::RowMajor> dhC_deltaTSC;

  dhC_projIntrinsic.setZero();
  dhC_distortion.setZero();
  double* dpC_deltaAll[] = {dhC_deltaTWS_full.data(),
                            dhC_deltahpW.data(),
                            dhC_deltaTSC_full.data(),
                            dhC_tr.data(),
                            dhC_td.data(),
                            dhC_sb.data(),
                            dhC_deltaTWS.data(),
                            dhC_deltaTSC.data()};
  RsReprojectionErrorAutoDiff<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL> rsre(*this);
  bool diffState =
          ::ceres::internal::AutoDifferentiate<
              ::ceres::internal::StaticParameterDims<7, 4, EXTRINSIC_MODEL::kGlobalDim, 1, 1, 9, 6,
              EXTRINSIC_MODEL::kNumParams>
             >(rsre, expandedParams, numOutputs, php_C, dpC_deltaAll);
  if (!diffState)
    std::cerr << "Potentially wrong Jacobians in autodiff " << std::endl;

  Eigen::VectorXd intrinsics(GEOMETRY_TYPE::NumIntrinsics);

  Eigen::Map<const Eigen::Matrix<double, PROJ_INTRINSIC_MODEL::kNumParams, 1>>
      projIntrinsics(parameters[3]);
  PROJ_INTRINSIC_MODEL::localToGlobal(projIntrinsics, &intrinsics);

  Eigen::Map<const Eigen::Matrix<double, kDistortionDim, 1>>
      distortionIntrinsics(parameters[4]);
  intrinsics.tail<kDistortionDim>() = distortionIntrinsics;
  cameraGeometryBase_->setIntrinsics(intrinsics);

  Eigen::Map<const Eigen::Vector4d> hp_C(&php_C[0]);

  // calculate the reprojection error
  measurement_t kp;
  Eigen::Matrix<double, 2, 4> Jh;
  Eigen::Matrix<double, 2, 4> Jh_weighted;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi;
  Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted;
  if (jacobians != NULL) {
    cameraGeometryBase_->projectHomogeneous(hp_C, &kp, &Jh, &Jpi);
    Jh_weighted = squareRootInformation_ * Jh;
    Jpi_weighted = squareRootInformation_ * Jpi;
  } else {
    cameraGeometryBase_->projectHomogeneous(hp_C, &kp);
  }

  measurement_t error = measurement_ - kp;

  // weight:
  measurement_t weighted_error = squareRootInformation_ * error;

  // assign:
  residuals[0] = weighted_error[0];
  residuals[1] = weighted_error[1];

  // check validity:
  bool valid = true;
  if (fabs(hp_C[3]) > 1.0e-8) {
    Eigen::Vector3d p_C = hp_C.template head<3>() / hp_C[3];
    if (p_C[2] < 0.2) {  // 20 cm - not very generic... but reasonable
      // std::cout<<"INVALID POINT"<<std::endl;
      valid = false;
    }
  }

  // calculate jacobians, if required
  // This is pretty close to Paul Furgale's thesis. eq. 3.100 on page 40
  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return true;
    }
    uint32_t height = cameraGeometryBase_->imageHeight();
    double ypixel(measurement_[1]);
    double kpN = ypixel / height - 0.5;
    assignJacobians(parameters, jacobians, jacobiansMinimal, Jh_weighted,
                    Jpi_weighted, dhC_deltaTWS, dhC_deltahpW, dhC_deltaTSC,
                    dhC_td, kpN, dhC_sb);
  }
  return true;
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
RsReprojectionErrorAutoDiff<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
RsReprojectionErrorAutoDiff(const RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>& rsre) :
    rsre_(rsre) {

}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL>
template <typename Scalar>
bool RsReprojectionErrorAutoDiff<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL>::
operator()(const Scalar* const T_WS, const Scalar* const php_W,
           const Scalar* const T_SC,
           const Scalar* const t_r,
           const Scalar* const t_d, const Scalar* const speedAndBiases,
           const Scalar* const deltaT_WS, const Scalar* const deltaT_SC,
           Scalar residuals[4]) const {
  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> t_WS_W0(T_WS);
  const Eigen::Quaternion<Scalar> q_WS0(T_WS[6], T_WS[3], T_WS[4], T_WS[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_WSe(deltaT_WS);
  Eigen::Matrix<Scalar, 3, 1> t_WS_W = t_WS_W0 + deltaT_WSe.template head<3>();
  Eigen::Matrix<Scalar, 3, 1> omega = deltaT_WSe.template tail<3>();
  Eigen::Quaternion<Scalar> dqWS = okvis::ceres::expAndTheta(omega);
  Eigen::Quaternion<Scalar> q_WS = dqWS * q_WS0;
  // q_WS.normalize();

  Eigen::Map<const Eigen::Matrix<Scalar, 4, 1>> hp_W(php_W);

  Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> t_SC_S0(T_SC);
  const Eigen::Quaternion<Scalar> q_SC0(T_SC[6], T_SC[3], T_SC[4], T_SC[5]);
  Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_SCe(deltaT_SC);
  Eigen::Matrix<Scalar, 3, 1> t_SC_S = t_SC_S0 + deltaT_SCe.template head<3>();
  omega = deltaT_SCe.template tail<3>();
  Eigen::Quaternion<Scalar> dqSC = okvis::ceres::expAndTheta(omega);
  Eigen::Quaternion<Scalar> q_SC = dqSC * q_SC0;

  Scalar trLatestEstimate = t_r[0];

  uint32_t height = rsre_.cameraGeometryBase_->imageHeight();
  double ypixel(rsre_.measurement_[1]);
  Scalar kpN = (Scalar)(ypixel / height - 0.5);
  Scalar tdLatestEstimate = t_d[0];
  Scalar relativeFeatureTime =
      tdLatestEstimate + trLatestEstimate * kpN - (Scalar)rsre_.tdAtCreation_;

  std::pair<Eigen::Quaternion<Scalar>, Eigen::Matrix<Scalar, 3, 1>> pairT_WS(
      q_WS, t_WS_W);
  Eigen::Matrix<Scalar, 9, 1> speedBgBa =
      Eigen::Map<const Eigen::Matrix<Scalar, 9, 1>>(speedAndBiases);

  Scalar t_start = (Scalar)rsre_.stateEpoch_.toSec();
  Scalar t_end = t_start + relativeFeatureTime;
  okvis::GenericImuMeasurementDeque<Scalar> imuMeasurements;
  for (size_t jack = 0; jack < rsre_.imuMeasCanopy_.size(); ++jack) {
    okvis::GenericImuMeasurement<Scalar> imuMeas(
        (Scalar)(rsre_.imuMeasCanopy_[jack].timeStamp.toSec()),
        rsre_.imuMeasCanopy_[jack].measurement.gyroscopes.template cast<Scalar>(),
        rsre_.imuMeasCanopy_[jack].measurement.accelerometers.template cast<Scalar>());
    imuMeasurements.push_back(imuMeas);
  }

  if (relativeFeatureTime >= Scalar(5e-8)) {
    okvis::ceres::predictStates(imuMeasurements, (Scalar)rsre_.gravityMag_, pairT_WS,
                                speedBgBa, t_start, t_end);
  } else if (relativeFeatureTime <= Scalar(-5e-8)) {
    okvis::ceres::predictStatesBackward(imuMeasurements, (Scalar)rsre_.gravityMag_,
                                        pairT_WS, speedBgBa, t_start, t_end);
  }

  q_WS = pairT_WS.first;
  t_WS_W = pairT_WS.second;

  // transform the point into the camera:
  Eigen::Matrix<Scalar, 3, 3> C_SC = q_SC.toRotationMatrix();
  Eigen::Matrix<Scalar, 3, 3> C_CS = C_SC.transpose();
  Eigen::Matrix<Scalar, 4, 4> T_CS = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_CS.template topLeftCorner<3, 3>() = C_CS;
  T_CS.template topRightCorner<3, 1>() = -C_CS * t_SC_S;
  Eigen::Matrix<Scalar, 3, 3> C_WS = q_WS.toRotationMatrix();
  Eigen::Matrix<Scalar, 3, 3> C_SW = C_WS.transpose();
  Eigen::Matrix<Scalar, 4, 4> T_SW = Eigen::Matrix<Scalar, 4, 4>::Identity();
  T_SW.template topLeftCorner<3, 3>() = C_SW;
  T_SW.template topRightCorner<3, 1>() = -C_SW * t_WS_W;
  Eigen::Matrix<Scalar, 4, 1> hp_S = T_SW * hp_W;
  Eigen::Matrix<Scalar, 4, 1> hp_C = T_CS * hp_S;

  residuals[0] = hp_C[0];
  residuals[1] = hp_C[1];
  residuals[2] = hp_C[2];
  residuals[3] = hp_C[3];

  return true;
}

}  // namespace ceres
}  // namespace okvis
