
#ifndef INCLUDE_OKVIS_CAMERAS_FOVDISTORTION_HPP_
#define INCLUDE_OKVIS_CAMERAS_FOVDISTORTION_HPP_

#include <Eigen/Core>
#include <memory>
#include "okvis/cameras/DistortionBase.hpp"

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief cameras Namespace for camera-related functionality.
namespace cameras {

class FovDistortion : public DistortionBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The default constructor with all zero ki
  inline FovDistortion();

  /// \brief Constructor initialising ki
  /// @param[in] w half FOV

  inline FovDistortion(double w);

  //////////////////////////////////////////////////////////////
  /// \name Methods related to generic parameters
  /// @{

  /// \brief set the generic parameters
  /// @param[in] parameters Parameter vector -- length must correspond
  /// numDistortionIntrinsics().
  /// @return    True if the requirements were followed.
  inline bool setParameters(const Eigen::VectorXd& parameters);

  /// \brief Obtain the generic parameters.
  inline bool getParameters(Eigen::VectorXd& parameters) const {
    parameters = parameters_;
    return true;
  }

  /// \brief The class type.
  inline std::string type() const { return "FovDistortion"; }

  /// \brief Number of distortion parameters
  inline int numDistortionIntrinsics() const { return NumDistortionIntrinsics; }

  static const int NumDistortionIntrinsics =
      1;  ///< The Number of distortion parameters.
  /// @}

  /// \brief Unit test support -- create a test distortion object
  static std::shared_ptr<DistortionBase> createTestObject() {
    return std::shared_ptr<DistortionBase>(new FovDistortion(1.0));
  }
  /// \brief Unit test support -- create a test distortion object
  static FovDistortion testObject() { return FovDistortion(1.0); }

  //////////////////////////////////////////////////////////////
  /// \name Distortion functions
  /// @{

  /// \brief Distortion only
  /// @param[in]  pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointDistorted   The distorted normalised (!) image point.
  /// @return     True on success (no singularity)
  inline bool distort(const Eigen::Vector2d& pointUndistorted,
                      Eigen::Vector2d* pointDistorted) const;

  /// \brief Distortion and Jacobians.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image
  /// point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the
  /// intrinsics vector.
  /// @return     True on success (no singularity)
  inline bool distort(const Eigen::Vector2d& pointUndistorted,
                      Eigen::Vector2d* pointDistorted,
                      Eigen::Matrix2d* pointJacobian,
                      Eigen::Matrix2Xd* parameterJacobian = NULL) const;

  /// \brief Distortion and Jacobians using external distortion intrinsics
  /// parameters.
  /// @param[in]  pointUndistorted  The undistorted normalised (!) image point.
  /// @param[in]  parameters        The distortion intrinsics vector.
  /// @param[out] pointDistorted    The distorted normalised (!) image point.
  /// @param[out] pointJacobian     The Jacobian w.r.t. changes on the image
  /// point.
  /// @param[out] parameterJacobian The Jacobian w.r.t. changes on the
  /// intrinsics vector.
  /// @return     True on success (no singularity)
  inline bool distortWithExternalParameters(
      const Eigen::Vector2d& pointUndistorted,
      const Eigen::VectorXd& parameters, Eigen::Vector2d* pointDistorted,
      Eigen::Matrix2d* pointJacobian = NULL,
      Eigen::Matrix2Xd* parameterJacobian = NULL) const;
  /// @}

  //////////////////////////////////////////////////////////////
  /// \name Undistortion functions
  /// @{

  /// \brief Undistortion only
  /// @param[in]  pointDistorted   The distorted normalised (!) image point.
  /// @param[out] pointUndistorted The undistorted normalised (!) image point.
  /// @return     True on success (no singularity)
  inline bool undistort(const Eigen::Vector2d& pointDistorted,
                        Eigen::Vector2d* pointUndistorted) const;

  /// \brief Undistortion only
  /// @param[in]  pointDistorted   The distorted normalised (!) image point.
  /// @param[out] pointUndistorted The undistorted normalised (!) image point.
  /// @param[out] pointJacobian    The Jacobian w.r.t. changes on the image
  /// point.
  /// @return     True on success (no singularity)
  inline bool undistort(const Eigen::Vector2d& pointDistorted,
                        Eigen::Vector2d* pointUndistorted,
                        Eigen::Matrix2d* pointJacobian) const;
  /// @}

 protected:
  Eigen::Matrix<double, NumDistortionIntrinsics, 1>
      parameters_;  ///< all distortion parameters

  double w_;
  static constexpr double kMaxValidAngle = (89.0 * M_PI / 180.0);
};

}  // namespace cameras
}  // namespace okvis

#include "implementation/FovDistortion.hpp"

#endif /* INCLUDE_OKVIS_CAMERAS_FOVDISTORTION_HPP_ */
