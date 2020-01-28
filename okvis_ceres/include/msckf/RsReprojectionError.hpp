
/**
 * @file ceres/RsReprojectionError.hpp
 * @brief Header file for the RsReprojectionError class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_MSCKF_RS_REPROJECTION_ERROR_HPP_
#define INCLUDE_MSCKF_RS_REPROJECTION_ERROR_HPP_

#include <vector>
#include <memory>
#include <ceres/ceres.h>
#include <okvis/Measurements.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

#include <msckf/ImuModels.hpp>
#include <msckf/ExtrinsicModels.hpp>
#include <msckf/PointLandmarkModels.hpp>

namespace okvis {
namespace ceres {

template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL, class EXTRINSIC_MODEL,
          class LANDMARK_MODEL, class IMU_MODEL>
class LocalBearingVector;

/// \brief The 2D keypoint reprojection error accounting for rolling shutter
///     skew and time offset and camera intrinsics.
/// \tparam GEOMETRY_TYPE The camera gemetry type.
/// \tparam PROJ_INTRINSIC_MODEL describes which subset of the projection
///     intrinsic parameters of the camera geometry model is represented and
///     optimized in the ceres solver.
///     It maps the subset to the full projection intrinsic parameters
///     using additional constant values from a provided camera geometry.
///     Its kNumParams should not be zero.
/// \tparam EXTRINSIC_MODEL describes which subset of the extrinsic parameters,
///     is represented and optimized in the ceres solver.
///     It maps the subset to the full extrinsic parameters using additional
///     constant values from a provided extrinsic entity, e.g., T_BC.
///     Its kNumParams should not be zero.
template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL=Extrinsic_p_BC_q_BC,
          class LANDMARK_MODEL=msckf::HomogeneousPointParameterization,
          class IMU_MODEL=Imu_BG_BA>
class RsReprojectionError
    : public ::ceres::SizedCostFunction<
          2 /* number of residuals */, 7 /* pose */, 4 /* landmark */,
          7 /* variable dim of extrinsics */,
          PROJ_INTRINSIC_MODEL::kNumParams /* variable dim of proj intrinsics
                                              (e.g., f, cx, cy) */,
          GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics,
          1 /* frame readout time */,
          1 /* time offset between visual and inertial data */,
          9 /* velocity and biases */
          >,
      public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Make the camera geometry type accessible.
  typedef GEOMETRY_TYPE camera_geometry_t;

  static const int kDistortionDim = GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics;

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<
      2, 7, 4, 7, PROJ_INTRINSIC_MODEL::kNumParams,
      GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics, 1, 1, 9>
      base_t;

  typedef typename std::conditional<
      (PROJ_INTRINSIC_MODEL::kNumParams > 1),
      Eigen::Matrix<double, 2, PROJ_INTRINSIC_MODEL::kNumParams,
                    Eigen::RowMajor>,
      Eigen::Matrix<double, 2, PROJ_INTRINSIC_MODEL::kNumParams> >::type
      ProjectionIntrinsicJacType;

  typedef typename std::conditional<
      (PROJ_INTRINSIC_MODEL::kNumParams > 1),
      Eigen::Matrix<double, 4, PROJ_INTRINSIC_MODEL::kNumParams,
                    Eigen::RowMajor>,
      Eigen::Matrix<double, 4, PROJ_INTRINSIC_MODEL::kNumParams>>::type
      ProjectionIntrinsicJacType4;

  typedef typename std::conditional<(kDistortionDim > 1),
      Eigen::Matrix<double, 2, kDistortionDim, Eigen::RowMajor>,
      Eigen::Matrix<double, 2, kDistortionDim>>::type DistortionJacType;

  /// \brief Number of residuals (2)
  static const int kNumResiduals = 2;

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d keypoint_t;

  /// \brief Measurement type (2D).
  typedef Eigen::Vector2d measurement_t;

  /// \brief Covariance / information matrix type (2x2).
  typedef Eigen::Matrix2d covariance_t;

  /// \brief Default constructor.
  RsReprojectionError();

  /**
   * @brief RsReprojectionError Construct with measurement and information matrix
   * @param cameraGeometry
   * @warning The camera geometry will be modified in evaluating Jacobians.
   * @param cameraId The id of the camera in the okvis::cameras::NCameraSystem.
   * @param measurement
   * @param information The information (weight) matrix.
   * @param imuMeasCanopy imu meas in the neighborhood of stateEpoch for
   *     compensating the rolling shutter effect.
   * @param stateEpoch epoch of the pose state and speed and biases
   * @param tdAtCreation the time offset at the creation of the state.
   * @param gravityMag
   */
  RsReprojectionError(
      std::shared_ptr<const camera_geometry_t> cameraGeometry,
      const measurement_t& measurement,
      const covariance_t& information,
      std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasCanopy,
      okvis::Time stateEpoch, double tdAtCreation, double gravityMag);

  /// \brief Trivial destructor.
  virtual ~RsReprojectionError()
  {
  }

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  virtual void setMeasurement(const measurement_t& measurement)
  {
    measurement_ = measurement;
  }

  /// \brief Set the underlying camera model.
  /// @param[in] cameraGeometry The camera geometry.
  void setCameraGeometry(
      std::shared_ptr<const camera_geometry_t> cameraGeometry)
  {
    cameraGeometryBase_ = cameraGeometry;
  }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  virtual void setInformation(const covariance_t& information);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  virtual const measurement_t& measurement() const
  {
    return measurement_;
  }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  virtual const covariance_t& information() const
  {
    return information_;
  }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  virtual const covariance_t& covariance() const
  {
    return covariance_;
  }

  // error term and Jacobian implementation
  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const * parameters, double* residuals,
                        double** jacobians) const;

  /**
   * @brief This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  bool EvaluateWithMinimalJacobiansAnalytic(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  bool EvaluateWithMinimalJacobiansGlobalAutoDiff(double const* const * parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  void assignJacobians(
      double const* const* parameters, double** jacobians,
      double** jacobiansMinimal, const Eigen::Matrix<double, 2, 4>& Jh_weighted,
      const Eigen::Matrix<double, 2, Eigen::Dynamic>& Jpi_weighted,
      const Eigen::Matrix<double, 4, 6>& dhC_deltaTWS,
      const Eigen::Matrix<double, 4, 4>& dhC_deltahpW,
      const Eigen::Matrix<double, 4, EXTRINSIC_MODEL::kNumParams>& dhC_dExtrinsic,
      const Eigen::Vector4d& dhC_td, double kpN,
      const Eigen::Matrix<double, 4, 9>& dhC_sb) const;

  void setJacobiansZero(double** jacobians, double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const
  {
    return kNumResiduals;
  }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const
  {
    return base_t::parameter_block_sizes().size();
  }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const
  {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Residual block type as string
  virtual std::string typeInfo() const
  {
    return "RsReprojectionError";
  }

  friend class LocalBearingVector<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>;
 protected:
//  uint64_t cameraId_; ///< ID of the camera.
  measurement_t measurement_; ///< The (2D) measurement.

  /// Warn: cameraGeometryBase_ may be updated with
  /// a ceres EvaluationCallback prior to Evaluate().
  // The camera model shared by all RsReprojectionError.
  std::shared_ptr<const camera_geometry_t> cameraGeometryBase_;

  // const after initialization
  std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasCanopy_;
  // weighting related
  covariance_t information_; ///< The 2x2 information matrix.
  covariance_t squareRootInformation_; ///< The 2x2 square root information matrix.
  covariance_t covariance_; ///< The 2x2 covariance matrix.

  okvis::Time stateEpoch_; ///< The timestamp of the set of robot states related to this error term.
  double tdAtCreation_; /// time offset at the creation of the states
  const double gravityMag_; ///< gravity in the world frame is [0, 0, -gravityMag_].
};

// For testing only
// Calculate the Jacobians of the homogeneous landmark coordinates in the
// camera frame, hp_C, relative to the states.
// AutoDifferentiate will invoke Evaluate() if the Functor is a ceres::CostFunction
// see ceres-solver/include/ceres/internal/variadic_evaluate.h
// so we have to separate operator() from Evaluate()
template <class GEOMETRY_TYPE, class PROJ_INTRINSIC_MODEL,
          class EXTRINSIC_MODEL, class LANDMARK_MODEL, class IMU_MODEL>
class LocalBearingVector {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LocalBearingVector(const RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>& rsre);
  template <typename Scalar>
  bool operator()(const Scalar* const T_WS, const Scalar* const hp_W,
                  const Scalar* const extrinsic, const Scalar* const t_r,
                  const Scalar* const t_d, const Scalar* const speedAndBiases,
                  const Scalar* const deltaT_WS,
                  const Scalar* const deltaExtrinsic, Scalar* hp_C) const;

 private:
  const RsReprojectionError<GEOMETRY_TYPE, PROJ_INTRINSIC_MODEL, EXTRINSIC_MODEL, LANDMARK_MODEL, IMU_MODEL>& rsre_;
};

}  // namespace ceres
}  // namespace okvis

#include "implementation/RsReprojectionError.hpp"
#endif /* INCLUDE_MSCKF_RS_REPROJECTION_ERROR_HPP_ */
