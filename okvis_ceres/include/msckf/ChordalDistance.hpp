
/**
 * @file ceres/ChordalDistance.hpp
 * @brief Header file for the ChordalDistance class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_MSCKF_CHORDAL_DISTANCE_HPP_
#define INCLUDE_MSCKF_CHORDAL_DISTANCE_HPP_

#include <vector>
#include <memory>
#include <ceres/ceres.h>
#include <okvis/Measurements.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ErrorInterface.hpp>

#include <msckf/ImuModels.hpp>
#include <msckf/ExtrinsicModels.hpp>
#include <msckf/PointSharedData.hpp>
#include <msckf/PointLandmarkModels.hpp>

namespace okvis {
namespace ceres {

/// \brief The chordal distance (N_{i,j} - R_{C(t_{i,j})} * f_{i,j}) accounting
/// for rolling shutter skew and time offset and camera intrinsics.
/// \warning A potential problem with this error term happens when
///     the provided IMU measurements do not cover camera observations to the
///     extent of the rolling shutter effect. This is most likely to occur with
///     observations in the most recent frame.
///     Because MSCKF uses observations up to the second most recent frame,
///     this problem should only happen to optimization-based estimator with
///     undelayed observations.
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
          class EXTRINSIC_MODEL = Extrinsic_p_BC_q_BC,
          class LANDMARK_MODEL = msckf::ParallaxAngleParameterization,
          class IMU_MODEL = Imu_BG_BA>
class ChordalDistance
    : public ::ceres::SizedCostFunction<
          3 /* residuals */, 7 /* observing frame pose */, 7 /* main anchor */,
          7 /* associate anchor */, LANDMARK_MODEL::kGlobalDim /* landmark */,
          7 /* camera extrinsic */,
          PROJ_INTRINSIC_MODEL::kNumParams /* camera projection intrinsic */,
          GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics,
          1 /* frame readout time */, 1 /* camera time delay */,
          9 /* velocity and biases of observing frame */,
          9 /* velocity and biases of main anchor */,
          9 /* velocity and biases of associate anchor */>,
      public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Make the camera geometry type accessible.
  typedef GEOMETRY_TYPE camera_geometry_t;
  static const int kProjectionIntrinsicDim = PROJ_INTRINSIC_MODEL::kNumParams;
  static const int kDistortionDim = GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics;
  static const int kNumResiduals = 3;
  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<
      kNumResiduals, 7, 7, 7, LANDMARK_MODEL::kGlobalDim, 7, PROJ_INTRINSIC_MODEL::kNumParams,
      GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics, 1, 1, 9, 9, 9>
      base_t;

  typedef typename std::conditional<
      (PROJ_INTRINSIC_MODEL::kNumParams > 1),
      Eigen::Matrix<double, kNumResiduals, PROJ_INTRINSIC_MODEL::kNumParams,
                    Eigen::RowMajor>,
      Eigen::Matrix<double, kNumResiduals, PROJ_INTRINSIC_MODEL::kNumParams> >::type
      ProjectionIntrinsicJacType;

  typedef typename std::conditional<(kDistortionDim > 1),
      Eigen::Matrix<double, kNumResiduals, kDistortionDim, Eigen::RowMajor>,
      Eigen::Matrix<double, kNumResiduals, kDistortionDim>>::type DistortionJacType;

  /// \brief Default constructor.
  ChordalDistance();

  /**
   * @brief ChordalDistance Construct with measurement and information matrix
   * @param cameraGeometry
   * @warning The camera geometry will be modified in evaluating Jacobians.
   * @param cameraId The id of the camera in the okvis::cameras::NCameraSystem.
   * @param measurement
   * @param information The information (weight) matrix.
   * @param pointDataPtr shared data of the landmark to compute propagated
   * poses and velocities at observation epochs.
   */
  ChordalDistance(
      std::shared_ptr<const camera_geometry_t> cameraGeometry,
      const Eigen::Vector2d& imageObservation,
      const Eigen::Matrix2d& observationCovariance,
      std::shared_ptr<const msckf::PointSharedData> pointDataPtr);

  /// \brief Trivial destructor.
  virtual ~ChordalDistance()
  {
  }

  /// \brief Set the underlying camera model.
  /// @param[in] cameraGeometry The camera geometry.
  void setCameraGeometry(
      std::shared_ptr<const camera_geometry_t> cameraGeometry)
  {
    cameraGeometryBase_ = cameraGeometry;
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
    return "ChordalDistance";
  }

 protected:
  Eigen::Vector2d measurement_; ///< The image observation.

  /// Warn: cameraGeometryBase_ may be updated with
  /// a ceres EvaluationCallback prior to Evaluate().
  std::shared_ptr<const camera_geometry_t> cameraGeometryBase_;

  // weighting related
  Eigen::Matrix2d observationCovariance_;
  mutable Eigen::Matrix3d squareRootInformation_; // updated in Evaluate()
  mutable Eigen::Matrix3d covariance_;

  int observationIndex_; ///< Index of the observation in the map point shared data.
  std::shared_ptr<const msckf::PointSharedData> pointDataPtr_;
};

}  // namespace ceres
}  // namespace okvis

#include "implementation/ChordalDistance.hpp"
#endif /* INCLUDE_MSCKF_CHORDAL_DISTANCE_HPP_ */
