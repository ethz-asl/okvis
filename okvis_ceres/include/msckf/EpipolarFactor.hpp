
/**
 * @file ceres/EpipolarFactor.hpp
 * @brief Header file for the EpipolarFactor class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_EPIPOLAR_FACTOR_HPP_
#define INCLUDE_OKVIS_CERES_EPIPOLAR_FACTOR_HPP_

#include <vector>
#include <memory>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <ceres/ceres.h>
#include <okvis/Measurements.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/ceres/PoseLocalParameterization.hpp>
#include <okvis/ceres/ReprojectionErrorBase.hpp>

namespace okvis {
namespace ceres {

/**
 * \brief The 1D epipolar error.
 * \tparam GEOMETRY_TYPE The camera gemetry type.
 * The constant params are passed into the residual error through the constructor interface.
 * The variable params are reflected in terms of dim in the SizedCostFunction base class.
 * The Jacobians are computed according to these dims except for the reparameterized pose.
 * The variable params will be passed to the evaluate function as scalar
 * pointers so they can be stored as vector<scalar> or Eigen::Vector.
 */
template <class GEOMETRY_TYPE, class EXTRINSIC_MODEL,
          class PROJ_INTRINSIC_MODEL>
class EpipolarFactor
    : public ::ceres::SizedCostFunction<
          1 /* residuals */, 7 /* left pose */, 7 /* right pose */,
          7 /* extrinsics */,
          PROJ_INTRINSIC_MODEL::kNumParams /* projecction intrinsics */,
          GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics,
          1 /* readout time */,
          1 /* camera time delay */>,
      public ReprojectionErrorBase /* use this base to simplify handling visual
                                      constraints in marginalization. */
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  /// \brief Make the camera geometry type accessible.
  typedef GEOMETRY_TYPE camera_geometry_t;

  static const int kDistortionDim = GEOMETRY_TYPE::distortion_t::NumDistortionIntrinsics;

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<
      1, 7, 7, 7, PROJ_INTRINSIC_MODEL::kNumParams,
      kDistortionDim, 1, 1>
      base_t;

  /// \brief Number of residuals (2)
  static const int kNumResiduals = 1;

  /// \brief The keypoint type (measurement type).
  typedef Eigen::Vector2d keypoint_t;

  /// \brief Measurement type (1D).
  typedef double measurement_t;

  /// \brief Covariance / information matrix type.
  typedef double covariance_t;

  /// \brief Default constructor.
  EpipolarFactor();

  /**
   * @brief EpipolarFactor Construct with measurement and information matrix
   * @param cameraGeometry
   * @warning The camera geometry will be modified in evaluating Jacobians.
   * @param measurement12 left and right 2d measurements
   * @param covariance12 left and right 2d covariance for 2d meas
   * @param imuMeasCanopy imu measurements in neighborhoods of the left and
   *     right stateEpochs
   * @param stateEpoch left and right state timestamps
   * @param tdAtCreation left and right reference td
   * @param gravityMag magnitude of gravity
   */
  EpipolarFactor(
      std::shared_ptr<camera_geometry_t> cameraGeometry,
      uint64_t landmarkId,
      const std::vector<Eigen::Vector2d,
                        Eigen::aligned_allocator<Eigen::Vector2d>>&
          measurement12,
      const std::vector<Eigen::Matrix2d,
                        Eigen::aligned_allocator<Eigen::Matrix2d>>&
          covariance12,
      std::vector<std::shared_ptr<const okvis::ImuMeasurementDeque>>&
          imuMeasCanopy,
      const std::vector<okvis::Time>& stateEpoch,
      const std::vector<double>& tdAtCreation,
      const std::vector<Eigen::Matrix<double, 9, 1>,
                        Eigen::aligned_allocator<Eigen::Matrix<double, 9, 1>>>&
          speedAndBiases,
      double gravityMag);

  /// \brief Trivial destructor.
  virtual ~EpipolarFactor()
  {
  }


  /// \brief Set the underlying camera model.
  /// @param[in] cameraGeometry The camera geometry.
  void setCameraGeometry(
      std::shared_ptr<camera_geometry_t> cameraGeometry)
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
    return "EpipolarFactor";
  }

 protected:

  // An camera model shared by all EpipolarFactor.
  // The camera model is volatile and updated in every Evaluate() step.
  mutable std::shared_ptr<camera_geometry_t> cameraGeometryBase_;

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> measurement_;
  std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>> covariance_;

  // const after initialization
  std::vector<std::shared_ptr<const okvis::ImuMeasurementDeque>> imuMeasCanopy_;

  // weighting related, they will be computed along with the residual
//  double information_; ///< The information matrix.
  mutable double squareRootInformation_; ///< The square root information matrix.

  std::vector<okvis::Time> stateEpoch_; ///< The timestamp of the set of robot states related to this error term.
  std::vector<double> tdAtCreation_;
  ///< To avoid complication in marginalizing speed and biases, first estimates
  ///  of speed and biases are used for computing pose at exposure.
  std::vector<Eigen::Matrix<double, 9, 1>,
              Eigen::aligned_allocator<Eigen::Matrix<double, 9, 1>>>
      speedAndBiases_;
  const double gravityMag_; ///< gravity in the world frame is [0, 0, -gravityMag_].
  double dtij_dtr_[2];  ///< kpN for left and right obs

  /**
   * @brief poseAndVelocityAtObservation compute T_WS nad velocity at observing a feature.
   * @param[in/out] pairT_WS in order to avoid use of okvis::Transformation
   * @param[in/out] velAndOmega records the linear and angular velocity compensated by the gyro bias
   * @param[in] parameters as used in Evaluate()
   * @param[in] index 0 for left camera, 1 for right
   */
  void poseAndVelocityAtObservation(
      std::pair<Eigen::Matrix<double, 3, 1>, Eigen::Quaternion<double>>*
          pairT_WS,
      Eigen::Matrix<double, 6, 1>* velAndOmega,
      double const* const* parameters, int index) const;
};

}  // namespace ceres
}  // namespace okvis

#include "implementation/EpipolarFactor.hpp"
#endif /* INCLUDE_OKVIS_CERES_EPIPOLAR_FACTOR_HPP_ */
