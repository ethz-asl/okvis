
/**
 * @file implementation/ChordalDistance.hpp
 * @brief Header implementation file for the ChordalDistance class.
 * @author Jianzhu Huai
 */
#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/MatrixPseudoInverse.hpp>
#include <okvis/kinematics/operators.hpp>

#include <msckf/DirectionFromParallaxAngleJacobian.hpp>
#include <msckf/EpipolarJacobian.hpp>
#include <msckf/JacobianHelpers.hpp>
#include <msckf/Measurements.hpp>
#include <msckf/imu/SimpleImuOdometry.hpp>
#include <msckf/imu/SimpleImuPropagationJacobian.hpp>
#include <msckf/TransformMultiplyJacobian.hpp>
#include <msckf/VectorNormalizationJacobian.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL,
                    EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::ChordalDistance() :
  R_WCnmf_(false) {}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    ChordalDistance(
        std::shared_ptr<const camera_geometry_t> cameraGeometry,
        const Eigen::Vector2d& imageObservation,
        const Eigen::Matrix2d& observationCovariance,
        int observationIndex,
        std::shared_ptr<const msckf::PointSharedData> pointDataPtr,
        bool R_WCnmf) :
    observationIndex_(observationIndex),
    pointDataPtr_(pointDataPtr),
    R_WCnmf_(R_WCnmf) {
  measurement_ = imageObservation;
  observationCovariance_ = observationCovariance;
  cameraGeometryBase_ = cameraGeometry;
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
bool ChordalDistance<
    GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL,
    IMU_MODEL>::EvaluateForMainAnchor(double const* const* parameters,
                                      double* residuals,
                                      double** jacobians,
                                      double** jacobiansMinimal) const {
  LWF::ParallaxAnglePoint pap;
  pap.set(parameters[3]);

  Eigen::Vector3d xy1;
  bool backProjectOk = cameraGeometryBase_->backProject(measurement_, &xy1);
  msckf::VectorNormalizationJacobian unit_fj_jacobian(xy1);
  Eigen::Vector3d unit_fj = unit_fj_jacobian.normalized();
  Eigen::Vector3d error = pap.n_.getVec() - unit_fj;
  // weight
  int projOptModelId = PROJ_INTRINSIC_MODEL::kModelId;
  Eigen::Matrix<double, 3, Eigen::Dynamic> dfj_dXcam;
  Eigen::Matrix3d cov_fj;
  bool projectOk =
      obsDirectionJacobian(xy1, cameraGeometryBase_, projOptModelId,
                           observationCovariance_, &dfj_dXcam, &cov_fj);
  Eigen::Matrix3d dunit_fj_dfj;
  unit_fj_jacobian.dxi_dvec(&dunit_fj_dfj);
  Eigen::Matrix<double, kNumResiduals, Eigen::Dynamic> dunit_fj_dXcam =
      dunit_fj_dfj * dfj_dXcam;
  covariance_ = dunit_fj_dfj * cov_fj * dunit_fj_dfj.transpose();
  Eigen::Matrix3d pinvCovSqrt;
  okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(
      covariance_, pinvCovSqrt, std::numeric_limits<double>::epsilon());
  squareRootInformation_.noalias() = pinvCovSqrt.transpose();
  Eigen::Vector3d weighted_error = squareRootInformation_ * error;
  // assign
  Eigen::Map<Eigen::Vector3d> resvec(residuals);
  resvec = weighted_error;
  bool valid = backProjectOk && projectOk;
  if (jacobians != NULL) {
    setJacobiansZero(jacobians, jacobiansMinimal);
    if (!valid) {
      return false;
    }
    // compute de/du, de/dxcam.
    if (jacobians[3]) {
      Eigen::Matrix3d jMinimal;
      jMinimal.topLeftCorner<3, 2>() = pap.n_.getM();
      jMinimal.col(2).setZero();
      jMinimal = (squareRootInformation_ * jMinimal).eval();
      Eigen::Matrix<double, LANDMARK_MODEL::kLocalDim,
                    LANDMARK_MODEL::kGlobalDim, Eigen::RowMajor>
          jLift;
      LANDMARK_MODEL::liftJacobian(parameters[3], jLift.data());
      Eigen::Map<Eigen::Matrix<double, kNumResiduals,
                               LANDMARK_MODEL::kGlobalDim, Eigen::RowMajor>>
          j(jacobians[3]);
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[3]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals,
                                   LANDMARK_MODEL::kLocalDim, Eigen::RowMajor>>
              jM(jacobiansMinimal[3]);
          jM = jMinimal;
        }
      }
    }
    if (jacobians[5]) {
      Eigen::Map<ProjectionIntrinsicJacType> j(jacobians[5]);
      j.noalias() = -squareRootInformation_ *
                    dunit_fj_dXcam.topLeftCorner<3, kProjectionIntrinsicDim>();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[5]) {
          Eigen::Map<ProjectionIntrinsicJacType> jM(jacobiansMinimal[5]);
          jM = j;
        }
      }
    }
    if (jacobians[6]) {
      Eigen::Map<DistortionJacType> j(jacobians[6]);
      j.noalias() = -squareRootInformation_ *
                    dunit_fj_dXcam.topRightCorner<3, kDistortionDim>();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[6]) {
          Eigen::Map<DistortionJacType> jM(jacobiansMinimal[6]);
          jM = j;
        }
      }
    }
  }
  return valid;
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
bool ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    EvaluateForMainAnchorRWC(double const* const* parameters, double* residuals,
             double** jacobians, double** jacobiansMinimal) const {
  LWF::ParallaxAnglePoint pap;
  pap.set(parameters[3]);

  Eigen::Matrix<double, 3, 1> t_BC_B(parameters[4][0], parameters[4][1],
                                     parameters[4][2]);
  Eigen::Quaternion<double> q_BC(parameters[4][6], parameters[4][3],
                                 parameters[4][4], parameters[4][5]);
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_BC(t_BC_B, q_BC);

  okvis::kinematics::Transformation T_WBtij =
      pointDataPtr_->T_WBtij(observationIndex_);
  msckf::TransformMultiplyJacobian T_WCtij_jacobian(
      std::make_pair(T_WBtij.r(), T_WBtij.q()), pair_T_BC);
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtij =
      T_WCtij_jacobian.multiply();

  Eigen::Vector3d xy1;
  bool backProjectOk = cameraGeometryBase_->backProject(measurement_, &xy1);
  msckf::VectorNormalizationJacobian unit_fj_jacobian(xy1);
  Eigen::Vector3d unit_fj = unit_fj_jacobian.normalized();
  Eigen::Matrix3d R_WCtij = pair_T_WCtij.second.toRotationMatrix();
  Eigen::Vector3d error = R_WCtij * (pap.n_.getVec() - unit_fj);
  // weight
  int projOptModelId = PROJ_INTRINSIC_MODEL::kModelId;
  Eigen::Matrix<double, 3, Eigen::Dynamic> dfj_dXcam;
  Eigen::Matrix3d cov_fj;
  bool projectOk =
      obsDirectionJacobian(xy1, cameraGeometryBase_, projOptModelId,
                           observationCovariance_, &dfj_dXcam, &cov_fj);
  Eigen::Matrix3d dunit_fj_dfj;
  unit_fj_jacobian.dxi_dvec(&dunit_fj_dfj);

  Eigen::Matrix<double, kNumResiduals, 3> de_dfj = -(R_WCtij * dunit_fj_dfj);
  Eigen::Matrix<double, kNumResiduals, Eigen::Dynamic> de_dXcam =
      de_dfj * dfj_dXcam;
  covariance_ = de_dfj * cov_fj * de_dfj.transpose();
  Eigen::Matrix3d pinvCovSqrt;
  okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(
      covariance_, pinvCovSqrt, std::numeric_limits<double>::epsilon());
  squareRootInformation_.noalias() = pinvCovSqrt.transpose();
  Eigen::Vector3d weighted_error = squareRootInformation_ * error;
  // assign
  Eigen::Map<Eigen::Vector3d> resvec(residuals);
  resvec = weighted_error;
  bool valid = backProjectOk && projectOk;
  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return false;
    }
    // use first estimates.
    okvis::kinematics::Transformation T_WBtij_forJac =
        pointDataPtr_->T_WBtij_ForJacobian(observationIndex_);

    msckf::TransformMultiplyJacobian T_WCtij_jacobian(
        std::make_pair(T_WBtij_forJac.r(), T_WBtij_forJac.q()), pair_T_BC);

    std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtij =
        T_WCtij_jacobian.multiply();
    Eigen::Matrix3d R_WCtij = pair_T_WCtij.second.toRotationMatrix();
    Eigen::Matrix3d de_dtheta_WCtij = -okvis::kinematics::crossMx(
          R_WCtij * (pap.n_.getVec() - unit_fj));
    Eigen::Matrix3d dtheta_WCtij_dtheta_WBtij, dtheta_WCtij_dtheta_BC;
    dtheta_WCtij_dtheta_WBtij = T_WCtij_jacobian.dtheta_dtheta_AB();
    dtheta_WCtij_dtheta_BC = T_WCtij_jacobian.dtheta_dtheta_BC();
    // T_WCtij
    if (jacobians[0]) {
      Eigen::Matrix<double, kNumResiduals, 6> jMinimal;
      jMinimal.leftCols<3>().setZero();
      jMinimal.rightCols<3>() = squareRootInformation_ * de_dtheta_WCtij * dtheta_WCtij_dtheta_WBtij;
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> jLift;
      PoseLocalParameterization::liftJacobian(parameters[0], jLift.data());
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> j(jacobians[0]);
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[0]) {
            Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>> jM(jacobiansMinimal[0]);
            jM = jMinimal;
        }
      }
    }
    // T_WCm
    if (jacobians[1]) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> j(jacobians[1]);
        j = Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>>(jacobians[0]);
      if (jacobiansMinimal) {
        if (jacobiansMinimal[1]) {
            Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>> jM(jacobiansMinimal[1]);
            jM = Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>>(jacobiansMinimal[0]);
        }
      }
    }
    // T_WCa
    zeroJacobian<7, 6, kNumResiduals>(2, jacobians, jacobiansMinimal);
    // Landmark point
    if (jacobians[3]) {
      Eigen::Matrix3d jMinimal;
      jMinimal.topLeftCorner<3, 2>() = R_WCtij * pap.n_.getM();
      jMinimal.col(2).setZero();
      jMinimal = (squareRootInformation_ * jMinimal).eval();
      Eigen::Matrix<double, LANDMARK_MODEL::kLocalDim,
                    LANDMARK_MODEL::kGlobalDim, Eigen::RowMajor>
          jLift;
      LANDMARK_MODEL::liftJacobian(parameters[3], jLift.data());
      Eigen::Map<Eigen::Matrix<double, kNumResiduals,
                               LANDMARK_MODEL::kGlobalDim, Eigen::RowMajor>>
          j(jacobians[3]);
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[3]) {
          Eigen::Map<
              Eigen::Matrix<double, kNumResiduals, LANDMARK_MODEL::kLocalDim,
                            Eigen::RowMajor>>
              jM(jacobiansMinimal[3]);
          jM = jMinimal;
        }
      }
    }
    // Extrinsic
    if (jacobians[4]) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7>> j(jacobians[4]);
      Eigen::Matrix<double, kNumResiduals, EXTRINSIC_MODEL::kNumParams>
          jMinimal;
      jMinimal.template leftCols<3>().setZero();
      Eigen::Matrix<double, EXTRINSIC_MODEL::kNumParams, 7, Eigen::RowMajor>
          jLift;
      switch (EXTRINSIC_MODEL::kModelId) {
        case Extrinsic_p_CB::kModelId:
          EXTRINSIC_MODEL::liftJacobian(parameters[4], jLift.data());
          break;
        case Extrinsic_p_BC_q_BC::kModelId:
        default:
          jMinimal.template rightCols<3>() = squareRootInformation_ * de_dtheta_WCtij *
                                    dtheta_WCtij_dtheta_BC;
          PoseLocalParameterization::liftJacobian(parameters[4],
                                                  jLift.data());
          break;
      }
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[4]) {
          Eigen::Map<
              Eigen::Matrix<double, kNumResiduals,
                            EXTRINSIC_MODEL::kNumParams, Eigen::RowMajor>>
              jM(jacobiansMinimal[4]);
          jM = jMinimal;
        }
      }
    }
    // projection intrinsic
    if (jacobians[5]) {
      Eigen::Map<ProjectionIntrinsicJacType> j(jacobians[5]);
      j.noalias() =
          squareRootInformation_ *
          de_dXcam.topLeftCorner<3, kProjectionIntrinsicDim>();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[5]) {
          Eigen::Map<ProjectionIntrinsicJacType> jM(jacobiansMinimal[5]);
          jM = j;
        }
      }
    }
    // distortion
    if (jacobians[6]) {
      Eigen::Map<DistortionJacType> j(jacobians[6]);
      j.noalias() = squareRootInformation_ *
                    de_dXcam.topRightCorner<3, kDistortionDim>();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[6]) {
          Eigen::Map<DistortionJacType> jM(jacobiansMinimal[6]);
          jM = j;
        }
      }
    }
    Eigen::Vector3d v_WBtij =
        pointDataPtr_->v_WBtij_ForJacobian(observationIndex_);
    Eigen::Vector3d omega_Btij = pointDataPtr_->omega_Btij(observationIndex_);
    T_WCtij_jacobian.setVelocity(v_WBtij, omega_Btij);
    Eigen::Vector3d dtheta_WCtij_dt = T_WCtij_jacobian.dtheta_dt();

    // readout time
    if (jacobians[7]) {
      double rowj = pointDataPtr_->normalizedRow(observationIndex_);
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> j(jacobians[7]);
      j = squareRootInformation_ * de_dtheta_WCtij * dtheta_WCtij_dt * rowj;
      if (jacobiansMinimal && jacobiansMinimal[7]) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> jM(
            jacobiansMinimal[7]);
        jM = j;
      }
    }
    // time offset
    if (jacobians[8]) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> j(jacobians[8]);
      j = squareRootInformation_ * de_dtheta_WCtij * dtheta_WCtij_dt;
      if (jacobiansMinimal && jacobiansMinimal[8]) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> jM(
            jacobiansMinimal[8]);
        jM = j;
      }
    }
    // v_WBtij and biases
    zeroJacobian<9, 9, kNumResiduals>(9, jacobians, jacobiansMinimal);
    // v_WBm and biases
    zeroJacobian<9, 9, kNumResiduals>(10, jacobians, jacobiansMinimal);
    // v_WBa and biases
    zeroJacobian<9, 9, kNumResiduals>(11, jacobians, jacobiansMinimal);
  }
  return valid;
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
bool ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    EvaluateWithMinimalJacobians(double const* const* parameters,
                                 double* residuals, double** jacobians,
                                 double** jacobiansMinimal) const {
  // We avoid the use of okvis::kinematics::Transformation here due to
  // quaternion normalization and so forth. This only matters in order to be
  // able to check Jacobians with numeric differentiation chained, first w.r.t.
  // q and then d_alpha.
  std::vector<int> anchorObservationIndices =
      pointDataPtr_->anchorObservationIds();
  if (anchorObservationIndices[0] == observationIndex_) {
    if (R_WCnmf_) {
      return EvaluateForMainAnchorRWC(parameters, residuals, jacobians, jacobiansMinimal);
    } else {
      return EvaluateForMainAnchor(parameters, residuals, jacobians, jacobiansMinimal);
    }
  }

  LWF::ParallaxAnglePoint pap;
  pap.set(parameters[3]);

  Eigen::Matrix<double, 3, 1> t_BC_B(parameters[4][0], parameters[4][1],
                                     parameters[4][2]);
  Eigen::Quaternion<double> q_BC(parameters[4][6], parameters[4][3],
                                 parameters[4][4], parameters[4][5]);
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_BC(t_BC_B, q_BC);

  okvis::kinematics::Transformation T_WBtij =
      pointDataPtr_->T_WBtij(observationIndex_);
  msckf::TransformMultiplyJacobian T_WCtij_jacobian(
      std::make_pair(T_WBtij.r(), T_WBtij.q()), pair_T_BC);
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtij =
      T_WCtij_jacobian.multiply();

  // compute N_{i,j}.
  okvis::kinematics::Transformation T_WBtmi =
      pointDataPtr_->T_WBtij(anchorObservationIndices[0]);
  okvis::kinematics::Transformation T_WBtai =
      pointDataPtr_->T_WBtij(anchorObservationIndices[1]);

  msckf::TransformMultiplyJacobian T_WCtmi_jacobian(
      std::make_pair(T_WBtmi.r(), T_WBtmi.q()), pair_T_BC);
  msckf::TransformMultiplyJacobian T_WCtai_jacobian(
      std::make_pair(T_WBtai.r(), T_WBtai.q()), pair_T_BC);

  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtmi =
      T_WCtmi_jacobian.multiply();
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtai =
      T_WCtai_jacobian.multiply();

  msckf::DirectionFromParallaxAngleJacobian NijFunction(
      pair_T_WCtmi, pair_T_WCtai.first, pair_T_WCtij.first, pap);
  Eigen::Vector3d Nij = NijFunction.evaluate();
  msckf::VectorNormalizationJacobian unit_Nij_jacobian(Nij);
  Eigen::Vector3d unit_Nij = unit_Nij_jacobian.normalized();

  // compute R_{WC(t_{i,j})} * f_{i,j}.
  Eigen::Vector3d xy1;
  bool backProjectOk = cameraGeometryBase_->backProject(measurement_, &xy1);
  msckf::VectorNormalizationJacobian unit_fj_jacobian(xy1);
  Eigen::Vector3d unit_fj = unit_fj_jacobian.normalized();
  Eigen::Vector3d error = unit_Nij - pair_T_WCtij.second * unit_fj;

  // weight and assign: compute Jacobians and covariance for the obs direction.
  Eigen::Matrix<double, 3, Eigen::Dynamic> dfj_dXcam;
  Eigen::Matrix3d cov_fj;
  int projOptModelId = PROJ_INTRINSIC_MODEL::kModelId;
  bool projectOk =
      obsDirectionJacobian(xy1, cameraGeometryBase_, projOptModelId,
                           observationCovariance_, &dfj_dXcam, &cov_fj);
  Eigen::Matrix3d dunit_fj_dfj;
  unit_fj_jacobian.dxi_dvec(&dunit_fj_dfj);
  Eigen::Matrix3d dRf_dfj =
      pair_T_WCtij.second.toRotationMatrix() * dunit_fj_dfj;
  covariance_ = dRf_dfj * cov_fj * dRf_dfj.transpose();
  Eigen::Matrix3d pinvCovSqrt;
  okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(
      covariance_, pinvCovSqrt, std::numeric_limits<double>::epsilon());
  squareRootInformation_.noalias() = pinvCovSqrt.transpose();
  Eigen::Vector3d weighted_error = squareRootInformation_ * error;
  Eigen::Map<Eigen::Vector3d> resvec(residuals);
  resvec = weighted_error;
  bool valid = backProjectOk && projectOk;

  // calculate jacobians, if required
  if (jacobians != NULL) {
    if (!valid) {
      setJacobiansZero(jacobians, jacobiansMinimal);
      return false;
    }
    // use first estimates.
    okvis::kinematics::Transformation T_WBtij_forJac =
        pointDataPtr_->T_WBtij_ForJacobian(observationIndex_);
    okvis::kinematics::Transformation T_WBtmi_forJac =
        pointDataPtr_->T_WBtij_ForJacobian(anchorObservationIndices[0]);
    okvis::kinematics::Transformation T_WBtai_forJac =
        pointDataPtr_->T_WBtij_ForJacobian(anchorObservationIndices[1]);

    msckf::TransformMultiplyJacobian T_WCtij_jacobian(
        std::make_pair(T_WBtij_forJac.r(), T_WBtij_forJac.q()), pair_T_BC);
    msckf::TransformMultiplyJacobian T_WCtmi_jacobian(
        std::make_pair(T_WBtmi_forJac.r(), T_WBtmi_forJac.q()), pair_T_BC);
    msckf::TransformMultiplyJacobian T_WCtai_jacobian(
        std::make_pair(T_WBtai_forJac.r(), T_WBtai_forJac.q()), pair_T_BC);
    std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtij =
        T_WCtij_jacobian.multiply();
    std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtmi =
        T_WCtmi_jacobian.multiply();
    std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_WCtai =
        T_WCtai_jacobian.multiply();
    msckf::DirectionFromParallaxAngleJacobian directionFromParallaxAngleJacobian(
        pair_T_WCtmi, pair_T_WCtai.first, pair_T_WCtij.first, pap);
    Eigen::Vector3d lP_Nij = directionFromParallaxAngleJacobian.evaluate();
    msckf::VectorNormalizationJacobian unit_Nij_jacobian(lP_Nij);
    Eigen::Matrix3d de_dN;
    unit_Nij_jacobian.dxi_dvec(&de_dN);
    Eigen::Matrix3d dN_dp_WCtij;
    directionFromParallaxAngleJacobian.dN_dp_WCtij(&dN_dp_WCtij);
    Eigen::Matrix3d dN_dtheta_WCtmi;
    directionFromParallaxAngleJacobian.dN_dtheta_WCmi(&dN_dtheta_WCtmi);
    Eigen::Matrix3d dN_dp_WCtmi;
    directionFromParallaxAngleJacobian.dN_dp_WCmi(&dN_dp_WCtmi);
    Eigen::Matrix3d dN_dp_WCtai;
    directionFromParallaxAngleJacobian.dN_dp_WCai(&dN_dp_WCtai);
    Eigen::Matrix<double, 3, 2> dN_dni;
    directionFromParallaxAngleJacobian.dN_dni(&dN_dni);
    Eigen::Matrix<double, 3, 1> dN_dthetai;
    directionFromParallaxAngleJacobian.dN_dthetai(&dN_dthetai);
    Eigen::Matrix3d dp_WCtij_dp_WBtij = T_WCtij_jacobian.dp_dp_AB();
    Eigen::Matrix3d dp_WCtmi_dp_WBtmi = T_WCtmi_jacobian.dp_dp_AB();
    Eigen::Matrix3d dp_WCtmi_dtheta_WBtmi = T_WCtmi_jacobian.dp_dtheta_AB();
    Eigen::Matrix3d dtheta_WCtmi_dtheta_WBtmi = T_WCtmi_jacobian.dtheta_dtheta_AB();
    Eigen::Matrix3d dp_WCtai_dp_WBtai = T_WCtai_jacobian.dp_dp_AB();
    // T_WBj
    if (jacobians[0]) {
      Eigen::Matrix<double, kNumResiduals, 6> jMinimal;
      Eigen::Matrix3d dtheta_dtheta_WBtij = T_WCtij_jacobian.dtheta_dtheta_AB();
      jMinimal.leftCols<3>() = de_dN * dN_dp_WCtij * dp_WCtij_dp_WBtij;
      if (anchorObservationIndices[1] == observationIndex_) {
        jMinimal.leftCols<3>() += de_dN * dN_dp_WCtai * dp_WCtai_dp_WBtai;
      }
      jMinimal.rightCols<3>() =
          okvis::kinematics::crossMx(pair_T_WCtij.second * unit_fj) *
          dtheta_dtheta_WBtij;
      jMinimal = (squareRootInformation_ * jMinimal).eval();
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> j(
          jacobians[0]);
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> jLift;
      PoseLocalParameterization::liftJacobian(parameters[0], jLift.data());
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[0]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>>
              jM(jacobiansMinimal[0]);
          jM = jMinimal;
        }
      }
    }
    // T_WBm
    if (jacobians[1]) {
      Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor> jMinimal;
      jMinimal.leftCols<3>() =
          squareRootInformation_ * de_dN * dN_dp_WCtmi * dp_WCtmi_dp_WBtmi;
      jMinimal.rightCols<3>() = squareRootInformation_ * de_dN *
                                (dN_dtheta_WCtmi * dtheta_WCtmi_dtheta_WBtmi +
                                 dN_dp_WCtmi * dp_WCtmi_dtheta_WBtmi);
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> j(
          jacobians[1]);
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> jLift;
      PoseLocalParameterization::liftJacobian(parameters[1], jLift.data());
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[1]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>>
              jM(jacobiansMinimal[1]);
          jM = jMinimal;
        }
      }
    }
    // T_WBa
    if (jacobians[2]) {
      Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor> jMinimal;
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> j(
          jacobians[2]);
      if (anchorObservationIndices[1] == observationIndex_) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> jj(
            jacobians[0]);
        j = jj;
      } else {
        jMinimal.leftCols(3) =
            squareRootInformation_ * de_dN * dN_dp_WCtai * dp_WCtai_dp_WBtai;
        jMinimal.rightCols(3).setZero();
        Eigen::Matrix<double, 6, 7, Eigen::RowMajor> jLift;
        PoseLocalParameterization::liftJacobian(parameters[2], jLift.data());
        j = jMinimal * jLift;
      }
      if (jacobiansMinimal) {
        if (jacobiansMinimal[2]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>>
              jM(jacobiansMinimal[2]);
          if (anchorObservationIndices[1] == observationIndex_) {
            Eigen::Map<Eigen::Matrix<double, kNumResiduals, 6, Eigen::RowMajor>>
                jjM(jacobiansMinimal[0]);
            jM = jjM;
          } else {
            jM = jMinimal;
          }
        }
      }
    }
    // point landmark
    if (jacobians[3]) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals,
                               LANDMARK_MODEL::kGlobalDim, Eigen::RowMajor>>
          j(jacobians[3]);
      Eigen::Matrix<double, 3, LANDMARK_MODEL::kLocalDim,
                    Eigen::RowMajor> jMinimal;
      jMinimal.template topLeftCorner<3, 2>() = dN_dni;
      jMinimal.col(2) = dN_dthetai;
      jMinimal = squareRootInformation_ * de_dN * jMinimal;
      Eigen::Matrix<double, LANDMARK_MODEL::kLocalDim,
                    LANDMARK_MODEL::kGlobalDim, Eigen::RowMajor>
          jLift;
      LANDMARK_MODEL::liftJacobian(parameters[3], jLift.data());
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[3]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals,
                                   LANDMARK_MODEL::kLocalDim, Eigen::RowMajor>>
              jM(jacobiansMinimal[3]);
          jM = jMinimal;
        }
      }
    }
    // T_BC
    if (jacobians[4]) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 7, Eigen::RowMajor>> j(
          jacobians[4]);
      Eigen::Matrix<double, kNumResiduals, EXTRINSIC_MODEL::kNumParams, Eigen::RowMajor> jMinimal;

      Eigen::Matrix3d dp_WCtmi_dp_BC = T_WCtmi_jacobian.dp_dp_BC();
      Eigen::Matrix3d dp_WCtai_dp_BC = T_WCtai_jacobian.dp_dp_BC();
      Eigen::Matrix3d dp_WCtij_dp_BC = T_WCtij_jacobian.dp_dp_BC();
      jMinimal.template leftCols<3>() =
          squareRootInformation_ * de_dN *
          (dN_dp_WCtmi * dp_WCtmi_dp_BC + dN_dp_WCtai * dp_WCtai_dp_BC +
           dN_dp_WCtij * dp_WCtij_dp_BC);
      Eigen::Matrix<double, EXTRINSIC_MODEL::kNumParams, 7, Eigen::RowMajor> jLift;
      switch (EXTRINSIC_MODEL::kModelId) {
        case Extrinsic_p_CB::kModelId:
          jMinimal.template leftCols<3>() = -jMinimal.template leftCols<3>() *
              pair_T_BC.second.toRotationMatrix();
          EXTRINSIC_MODEL::liftJacobian(parameters[4], jLift.data());
          break;
        case Extrinsic_p_BC_q_BC::kModelId:
        default:
          {

            Eigen::Matrix3d dtheta_WCtmi_dtheta_BC = T_WCtmi_jacobian.dtheta_dtheta_BC();
            Eigen::Matrix3d dp_WCtmi_dtheta_BC = T_WCtmi_jacobian.dp_dtheta_BC();
            Eigen::Matrix3d dp_WCtai_dtheta_BC = T_WCtai_jacobian.dp_dtheta_BC();
            Eigen::Matrix3d dp_WCtij_dtheta_BC = T_WCtij_jacobian.dp_dtheta_BC();
            Eigen::Matrix3d dtheta_WCtij_dtheta_BC = T_WCtij_jacobian.dtheta_dtheta_BC();
            jMinimal.template rightCols<3>() =
                squareRootInformation_ *
                (de_dN * (dN_dtheta_WCtmi * dtheta_WCtmi_dtheta_BC +
                          dN_dp_WCtmi * dp_WCtmi_dtheta_BC +
                          dN_dp_WCtai * dp_WCtai_dtheta_BC +
                          dN_dp_WCtij * dp_WCtij_dtheta_BC) +
                 okvis::kinematics::crossMx(pair_T_WCtij.second * unit_fj) *
                     dtheta_WCtij_dtheta_BC);
            PoseLocalParameterization::liftJacobian(parameters[4], jLift.data());
          }
          break;
      }
      j = jMinimal * jLift;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[4]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, EXTRINSIC_MODEL::kNumParams, Eigen::RowMajor>>
              jM(jacobiansMinimal[4]);
          jM = jMinimal;
        }
      }
    }
    // projection intrinsic
    if (jacobians[5]) {
      Eigen::Map<ProjectionIntrinsicJacType> j(jacobians[5]);
      j = -squareRootInformation_ * pair_T_WCtij.second.toRotationMatrix() *
          dunit_fj_dfj * dfj_dXcam.leftCols<kProjectionIntrinsicDim>();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[5]) {
          Eigen::Map<ProjectionIntrinsicJacType> jM(jacobiansMinimal[5]);
          jM = j;
        }
      }
    }
    // distortion
    if (jacobians[6]) {
      Eigen::Map<DistortionJacType> j(jacobians[6]);
      j = -squareRootInformation_ * pair_T_WCtij.second.toRotationMatrix() *
          dunit_fj_dfj * dfj_dXcam.rightCols<kDistortionDim>();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[6]) {
          Eigen::Map<DistortionJacType> jM(jacobiansMinimal[6]);
          jM = j;
        }
      }
    }
    // readout time
    Eigen::Vector3d v_WBtij =
        pointDataPtr_->v_WBtij_ForJacobian(observationIndex_);
    Eigen::Vector3d omega_Btij = pointDataPtr_->omega_Btij(observationIndex_);
    Eigen::Vector3d v_WBtmi =
        pointDataPtr_->v_WBtij_ForJacobian(anchorObservationIndices[0]);
    Eigen::Vector3d omega_Btmi =
        pointDataPtr_->omega_Btij(anchorObservationIndices[0]);
    Eigen::Vector3d v_WBtai =
        pointDataPtr_->v_WBtij_ForJacobian(anchorObservationIndices[1]);
    Eigen::Vector3d omega_Btai =
        pointDataPtr_->omega_Btij(anchorObservationIndices[1]);


    T_WCtmi_jacobian.setVelocity(v_WBtmi, omega_Btmi);
    Eigen::Vector3d dp_WCtmi_dt = T_WCtmi_jacobian.dp_dt();
    Eigen::Vector3d dtheta_WCtmi_dt = T_WCtmi_jacobian.dtheta_dt();
    T_WCtij_jacobian.setVelocity(v_WBtij, omega_Btij);
    Eigen::Vector3d dp_WCtij_dt = T_WCtij_jacobian.dp_dt();
    Eigen::Vector3d dtheta_WCtij_dt = T_WCtij_jacobian.dtheta_dt();
    T_WCtai_jacobian.setVelocity(v_WBtai, omega_Btai);
    Eigen::Vector3d dp_WCtai_dt = T_WCtai_jacobian.dp_dt();
    if (jacobians[7]) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> j(jacobians[7]);
      double rowj = pointDataPtr_->normalizedRow(observationIndex_);
      double rowm = pointDataPtr_->normalizedRow(anchorObservationIndices[0]);
      double rowa = pointDataPtr_->normalizedRow(anchorObservationIndices[1]);

      j = squareRootInformation_ * de_dN *
              (dN_dtheta_WCtmi * dtheta_WCtmi_dt * rowm +
               dN_dp_WCtmi * dp_WCtmi_dt * rowm +
               dN_dp_WCtai * dp_WCtai_dt * rowa +
               dN_dp_WCtij * dp_WCtij_dt * rowj) +
          squareRootInformation_ *
              okvis::kinematics::crossMx(pair_T_WCtij.second * unit_fj) *
              dtheta_WCtij_dt * rowj;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[7]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> jM(
              jacobiansMinimal[7]);
          jM = j;
        }
      }
    }
    // camera time delay
    if (jacobians[8]) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> j(jacobians[8]);
      j = squareRootInformation_ * de_dN *
              (dN_dtheta_WCtmi * dtheta_WCtmi_dt + dN_dp_WCtmi * dp_WCtmi_dt +
               dN_dp_WCtai * dp_WCtai_dt + dN_dp_WCtij * dp_WCtij_dt) +
          squareRootInformation_ *
              okvis::kinematics::crossMx(pair_T_WCtij.second * unit_fj) *
              dtheta_WCtij_dt;
      if (jacobiansMinimal) {
        if (jacobiansMinimal[8]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 1>> jM(
              jacobiansMinimal[8]);
          jM = j;
        }
      }
    }
    // speed and biases for observing frame.
    if (jacobians[9]) {
      double featureTime =
          pointDataPtr_->normalizedFeatureTime(observationIndex_);
      Eigen::Matrix3d de_dv_WBj =
          de_dN * dN_dp_WCtij * dp_WCtij_dp_WBtij * featureTime;
      if (observationIndex_ == anchorObservationIndices[1]) {
        de_dv_WBj += de_dN * dN_dp_WCtai * dp_WCtai_dp_WBtai * featureTime;
      }
      de_dv_WBj = squareRootInformation_ * de_dv_WBj;
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> j(
          jacobians[9]);
      j.leftCols(3) = de_dv_WBj;
      j.rightCols(6).setZero();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[9]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>>
              jM(jacobiansMinimal[9]);
          jM = j;
        }
      }
    }
    // speed and biases for main anchor.
    if (jacobians[10]) {
      double featureTime =
          pointDataPtr_->normalizedFeatureTime(anchorObservationIndices[0]);
      Eigen::Matrix3d de_dv_WBm = squareRootInformation_ * de_dN * dN_dp_WCtmi *
                                  dp_WCtmi_dp_WBtmi * featureTime;
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> j(
          jacobians[10]);
      j.leftCols(3) = de_dv_WBm;
      j.rightCols(6).setZero();
      if (jacobiansMinimal) {
        if (jacobiansMinimal[10]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>>
              jM(jacobiansMinimal[10]);
          jM = j;
        }
      }
    }
    // speed and biases for associate anchor.
    if (jacobians[11]) {
      Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> j(
          jacobians[11]);
      if (observationIndex_ == anchorObservationIndices[1]) {
        Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>> jj(
            jacobians[9]);
        j = jj;
      } else {
        double featureTime =
            pointDataPtr_->normalizedFeatureTime(anchorObservationIndices[1]);
        Eigen::Matrix3d de_dv_WBa = squareRootInformation_ * de_dN *
                                    dN_dp_WCtai * dp_WCtai_dp_WBtai *
                                    featureTime;
        j.leftCols(3) = de_dv_WBa;
        j.rightCols(6).setZero();
      }
      if (jacobiansMinimal) {
        if (jacobiansMinimal[11]) {
          Eigen::Map<Eigen::Matrix<double, kNumResiduals, 9, Eigen::RowMajor>>
              jM(jacobiansMinimal[11]);
          jM = j;
        }
      }
    }
  }
  return valid;
}

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
void ChordalDistance<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>::
    setJacobiansZero(double** jacobians, double** jacobiansMinimal) const {
  zeroJacobian<7, 6, kNumResiduals>(0, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, kNumResiduals>(1, jacobians, jacobiansMinimal);
  zeroJacobian<7, 6, kNumResiduals>(2, jacobians, jacobiansMinimal);
  zeroJacobian<LANDMARK_MODEL::kGlobalDim, LANDMARK_MODEL::kLocalDim, kNumResiduals>(3, jacobians, jacobiansMinimal);
  zeroJacobian<7, EXTRINSIC_MODEL::kNumParams, kNumResiduals>(4, jacobians, jacobiansMinimal);
  zeroJacobian<PROJ_INTRINSIC_MODEL::kNumParams,
               PROJ_INTRINSIC_MODEL::kNumParams, kNumResiduals>(5, jacobians, jacobiansMinimal);
  zeroJacobian<kDistortionDim, kDistortionDim, kNumResiduals>(6, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, kNumResiduals>(7, jacobians, jacobiansMinimal);
  zeroJacobian<1, 1, kNumResiduals>(8, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, kNumResiduals>(9, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, kNumResiduals>(10, jacobians, jacobiansMinimal);
  zeroJacobian<9, 9, kNumResiduals>(11, jacobians, jacobiansMinimal);
}
}  // namespace ceres
}  // namespace okvis
