
/**
 * @file implementation/ChordalDistance.hpp
 * @brief Header implementation file for the ChordalDistance class.
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
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL,
                    EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::ChordalDistance() {}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    ChordalDistance(
        std::shared_ptr<const camera_geometry_t> cameraGeometry,
        const measurement_t& measurement,
        const covariance_t& information,
        std::shared_ptr<const msckf::PointSharedData> pointDataPtr) :
    pointDataPtr_(pointDataPtr) {
  setMeasurement(measurement);
  setInformation(information);
  setCameraGeometry(cameraGeometry);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
void ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    setInformation(const covariance_t& information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error
  // weighting
  Eigen::LLT<Eigen::Matrix2d> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
bool ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    Evaluate(double const* const* parameters, double* residuals,
             double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
bool ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  return EvaluateWithMinimalJacobiansAnalytic(parameters, residuals, jacobians,
                                              jacobiansMinimal);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
bool ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    EvaluateWithMinimalJacobiansAnalytic(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  // We avoid the use of okvis::kinematics::Transformation here due to
  // quaternion normalization and so forth. This only matters in order to be
  // able to check Jacobians with numeric differentiation chained, first w.r.t.
  // q and then d_alpha.

  Eigen::Map<const Eigen::Vector3d> t_WB_W0(parameters[0]);
  const Eigen::Quaterniond q_WB0(parameters[0][6], parameters[0][3],
                                 parameters[0][4], parameters[0][5]);

  // the point in world coordinates
  Eigen::Map<const Eigen::Vector4d> hp_W(&parameters[1][0]);

  Eigen::Matrix<double, 3, 1> t_BC_B(parameters[2][0], parameters[2][1], parameters[2][2]);
  Eigen::Quaternion<double> q_BC(parameters[2][6], parameters[2][3], parameters[2][4],
                                 parameters[2][5]);
  double trLatestEstimate = parameters[5][0];
  double tdLatestEstimate = parameters[6][0];

  double ypixel(measurement_[1]);
  uint32_t height = cameraGeometryBase_->imageHeight();
  double kpN = ypixel / height - 0.5;
  double relativeFeatureTime = tdLatestEstimate + trLatestEstimate * kpN - tdAtCreation_;
  std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> pairT_WB(
      t_WB_W0, q_WB0);
  Eigen::Matrix<double, 9, 1> speedBgBa =
      Eigen::Map<const Eigen::Matrix<double, 9, 1>>(parameters[7]);

  const okvis::Time t_start = stateEpoch_;
  const okvis::Time t_end = stateEpoch_ + okvis::Duration(relativeFeatureTime);
  const double wedge = 5e-8;
  if (relativeFeatureTime >= wedge) {
    okvis::ceres::predictStates(*imuMeasCanopy_, gravityMag_, pairT_WB,
                                speedBgBa, t_start, t_end);
  } else if (relativeFeatureTime <= -wedge) {
    okvis::ceres::predictStatesBackward(*imuMeasCanopy_, gravityMag_, pairT_WB,
                                        speedBgBa, t_start, t_end);
  }

  Eigen::Quaterniond q_WB = pairT_WB.second;
  Eigen::Vector3d t_WB_W = pairT_WB.first;

  // transform the point into the camera:
  Eigen::Matrix3d C_BC = q_BC.toRotationMatrix();
  Eigen::Matrix3d C_CB = C_BC.transpose();
  Eigen::Matrix4d T_CB = Eigen::Matrix4d::Identity();
  T_CB.topLeftCorner<3, 3>() = C_CB;
  T_CB.topRightCorner<3, 1>() = -C_CB * t_BC_B;
  Eigen::Matrix3d C_WB = q_WB.toRotationMatrix();
  Eigen::Matrix3d C_BW = C_WB.transpose();
  Eigen::Matrix4d T_BW = Eigen::Matrix4d::Identity();
  T_BW.topLeftCorner<3, 3>() = C_BW;
  T_BW.topRightCorner<3, 1>() = -C_BW * t_WB_W;
  Eigen::Vector4d hp_B = T_BW * hp_W;
  Eigen::Vector4d hp_C = T_CB * hp_B;

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

  measurement_t error = kp - measurement_;

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
    std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>> lP_T_WB = pairT_WB;
    SpeedAndBiases lP_sb = speedBgBa;
    if (posVelAtLinearization_) {
      // compute p_WB, v_WB at (t_{f_i,j}) that use FIRST ESTIMATES of
      // position and velocity, i.e., their linearization point
      lP_T_WB = std::make_pair(posVelAtLinearization_->head<3>(), q_WB0);
      lP_sb = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(parameters[7]);
      lP_sb.head<3>() = posVelAtLinearization_->tail<3>();
      if (relativeFeatureTime >= wedge) {
        okvis::ceres::predictStates(*imuMeasCanopy_, gravityMag_, lP_T_WB,
                                    lP_sb, t_start, t_end);
      } else if (relativeFeatureTime <= -wedge) {
        okvis::ceres::predictStatesBackward(*imuMeasCanopy_, gravityMag_, lP_T_WB,
                                            lP_sb, t_start, t_end);
      }
      C_BW = lP_T_WB.second.toRotationMatrix().transpose();
      t_WB_W = lP_T_WB.first;
      T_BW.topLeftCorner<3, 3>() = C_BW;
      T_BW.topRightCorner<3, 1>() = -C_BW * t_WB_W;
      hp_B = T_BW * hp_W;
      hp_C = T_CB * hp_B;
    }

    Eigen::Matrix<double, 4, 6> dhC_deltaTWS;
    Eigen::Matrix<double, 4, 4> dhC_deltahpW;
    Eigen::Matrix<double, 4, EXTRINSIC_MODEL::kNumParams> dhC_dExtrinsic;
    Eigen::Vector4d dhC_td;
    Eigen::Matrix<double, 4, 9> dhC_sb;

    Eigen::Vector3d p_BP_W = hp_W.head<3>() - t_WB_W * hp_W[3];
    Eigen::Matrix<double, 4, 6> dhS_deltaTWS;
    dhS_deltaTWS.topLeftCorner<3, 3>() = -C_BW * hp_W[3];
    dhS_deltaTWS.topRightCorner<3, 3>() =
        C_BW * okvis::kinematics::crossMx(p_BP_W);
    dhS_deltaTWS.row(3).setZero();
    dhC_deltaTWS = T_CB * dhS_deltaTWS;
    dhC_deltahpW = T_CB * T_BW;

    EXTRINSIC_MODEL::dhC_dExtrinsic_HPP(hp_C, C_CB, &dhC_dExtrinsic);

    okvis::ImuMeasurement queryValue;
    okvis::ceres::interpolateInertialData(*imuMeasCanopy_, t_end, queryValue);
    queryValue.measurement.gyroscopes -= lP_sb.segment<3>(3);
    Eigen::Vector3d p =
        okvis::kinematics::crossMx(queryValue.measurement.gyroscopes) *
            hp_B.head<3>() +
        C_BW * lP_sb.head<3>() * hp_W[3];
    dhC_td.head<3>() = -C_CB * p;
    dhC_td[3] = 0;

    Eigen::Matrix3d dhC_vW = -C_CB * C_BW * relativeFeatureTime * hp_W[3];
    Eigen::Matrix3d dhC_bg =
        -C_CB * C_BW *
        okvis::kinematics::crossMx(hp_W.head<3>() - hp_W[3] * t_WB_W) *
        relativeFeatureTime * q_WB0.toRotationMatrix();

    dhC_sb.row(3).setZero();
    dhC_sb.topRightCorner<3, 3>().setZero();
    dhC_sb.topLeftCorner<3, 3>() = dhC_vW;
    dhC_sb.block<3, 3>(0, 3) = dhC_bg;

    assignJacobians(parameters, jacobians, jacobiansMinimal, Jh_weighted,
                    Jpi_weighted, dhC_deltaTWS, dhC_deltahpW, dhC_dExtrinsic,
                    dhC_td, kpN, dhC_sb);
  }
  return true;
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
void ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, 2>(0, jacobians, jacobiansMinimal);
  zeroJacobian<4, 3, 2>(1, jacobians, jacobiansMinimal);
  zeroJacobian<7, EXTRINSIC_MODEL::kNumParams, 2>(2, jacobians, jacobiansMinimal);
  zeroJacobian<PROJ_INTRINSIC_MODEL::kNumParams,
               PROJ_INTRINSIC_MODEL::kNumParams, 2>(3, jacobians,
                                                    jacobiansMinimal);
  zeroJacobian<kDistortionDim, kDistortionDim, 2>(4, jacobians,
                                                  jacobiansMinimal);
  zeroJacobian<1, 1, 2>(5, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, 2>(6, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, 2>(7, jacobians, jacobiansMinimal);
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
void ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    assignJacobians(
        double const* const* parameters, double** jacobians,
        double** jacobiansMinimal,
        const Eigen::Matrix<double, 2, 4>& Jh_weighted,
        const Eigen::Matrix<double, 2, Eigen::Dynamic>& Jpi_weighted,
        const Eigen::Matrix<double, 4, 6>& dhC_deltaTWS,
        const Eigen::Matrix<double, 4, 4>& dhC_deltahpW,
        const Eigen::Matrix<double, 4, EXTRINSIC_MODEL::kNumParams>& dhC_dExtrinsic,
        const Eigen::Vector4d& dhC_td, double kpN,
        const Eigen::Matrix<double, 4, 9>& dhC_sb) const {
  if (jacobians[0] != NULL) {
    Eigen::Matrix<double, 2, 6, Eigen::RowMajor> J0_minimal;
    J0_minimal = Jh_weighted * dhC_deltaTWS;
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
    J1 = Jh_weighted * dhC_deltahpW;
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
    Eigen::Matrix<double, 2, EXTRINSIC_MODEL::kNumParams, Eigen::RowMajor>
        J2_minimal = Jh_weighted * dhC_dExtrinsic;
    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J2(jacobians[2]);
    Eigen::Matrix<double, EXTRINSIC_MODEL::kNumParams, 7, Eigen::RowMajor> J_lift;
    if (EXTRINSIC_MODEL::kNumParams == 6) {
      // Warn: This relates to the parameterization of
      // CameraSensorStates::T_SCi in addStates, and ReprojectionError liftJacobian
      PoseLocalParameterization::liftJacobian(parameters[2], J_lift.data());
      J2 = J2_minimal * J_lift;
    } else {
      EXTRINSIC_MODEL::liftJacobian(parameters[2], J_lift.data());
      J2 = J2_minimal * J_lift;
    }
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[2] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, EXTRINSIC_MODEL::kNumParams, Eigen::RowMajor>>
            J2_minimal_mapped(jacobiansMinimal[2]);
        J2_minimal_mapped = J2_minimal;
      }
    }
  }

  // camera intrinsics
  if (jacobians[3] != NULL) {
    Eigen::Map<ProjectionIntrinsicJacType> J1(jacobians[3]);
    Eigen::Matrix<double, 2, Eigen::Dynamic> Jpi_weighted_copy = Jpi_weighted;
    PROJ_INTRINSIC_MODEL::kneadIntrinsicJacobian(&Jpi_weighted_copy);
    J1 = Jpi_weighted_copy
        .template topLeftCorner<2, PROJ_INTRINSIC_MODEL::kNumParams>();
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[3] != NULL) {
        Eigen::Map<ProjectionIntrinsicJacType> J1_minimal_mapped(jacobiansMinimal[3]);
        J1_minimal_mapped = J1;
      }
    }
  }

  if (jacobians[4] != NULL) {
    Eigen::Map<DistortionJacType> J1(jacobians[4]);
    J1 = Jpi_weighted.template topRightCorner<2, kDistortionDim>();
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[4] != NULL) {
        Eigen::Map<DistortionJacType>
            J1_minimal_mapped(jacobiansMinimal[4]);
        J1_minimal_mapped = J1;
      }
    }
  }

  if (jacobians[5] != NULL) {
    Eigen::Map<Eigen::Matrix<double, 2, 1>> J1(jacobians[5]);
    J1 = Jh_weighted * dhC_td * kpN;
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
    J1 = Jh_weighted * dhC_td;
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
    J1 = Jh_weighted * dhC_sb;
    if (jacobiansMinimal != NULL) {
      if (jacobiansMinimal[7] != NULL) {
        Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>>
            J1_minimal_mapped(jacobiansMinimal[7]);
        J1_minimal_mapped = J1;
      }
    }
  }
}

}  // namespace ceres
}  // namespace okvis
