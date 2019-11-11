
/**
 * @file CameraTimeParamBlock.hpp
 * @brief Header file for the CameraTimeParamBlock class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_CAMERATIMEPARAMBLOCK_HPP_
#define INCLUDE_OKVIS_CERES_CAMERATIMEPARAMBLOCK_HPP_

#include <okvis/ceres/ParameterBlockSized.hpp>
//#include <okvis/kinematics/Transformation.hpp>
#include <Eigen/Core>
#include <okvis/Time.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
const int nTimeDim = 1, nTimeMinDim = 1;
typedef double TimeOffset;  // can represent either T_d or T_r

/// \brief Wraps the parameter block for camera time offset estimate
class CameraTimeParamBlock
    : public ParameterBlockSized<nTimeDim, nTimeMinDim, TimeOffset> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /// \brief The base class type.
  typedef ParameterBlockSized<nTimeDim, nTimeMinDim, TimeOffset> base_t;

  /// \brief The estimate type (9D vector).
  typedef TimeOffset estimate_t;

  /// \brief Default constructor (assumes not fixed).
  CameraTimeParamBlock();

  /// \brief Constructor with estimate and time.
  /// @param[in] timeOffset The fx,fy,cx,cy estimate.
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  CameraTimeParamBlock(const TimeOffset& timeOffset, uint64_t id,
                       const okvis::Time& timestamp);

  /// \brief Trivial destructor.
  virtual ~CameraTimeParamBlock();

  // setters
  /// @brief Set estimate of this parameter block.
  /// @param[in] timeOffset The estimate to set this to.
  virtual void setEstimate(const TimeOffset& timeOffset);

  /// \brief Set the time.
  /// @param[in] timestamp The timestamp of this state.
  void setTimestamp(const okvis::Time& timestamp) { timestamp_ = timestamp; }

  // getters
  /// @brief Get estimate.
  /// \return The estimate.
  virtual TimeOffset estimate() const;

  /// \brief Get the time.
  /// \return The timestamp of this state.
  okvis::Time timestamp() const { return timestamp_; }

  // minimal internal parameterization
  // x0_plus_Delta=Delta_Chi[+]x0
  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x0 Variable.
  /// @param[in] Delta_Chi Perturbation.
  /// @param[out] x0_plus_Delta Perturbed x.
  virtual void plus(const double* x0, const double* Delta_Chi,
                    double* x0_plus_Delta) const {
    Eigen::Map<const Eigen::Matrix<double, nTimeDim, 1> > x0_(x0);
    Eigen::Map<const Eigen::Matrix<double, nTimeDim, 1> > Delta_Chi_(Delta_Chi);
    Eigen::Map<Eigen::Matrix<double, nTimeDim, 1> > x0_plus_Delta_(
        x0_plus_Delta);
    x0_plus_Delta_ = x0_ + Delta_Chi_;
  }

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  //  /// @param[in] x0 Variable.
  /// @param[out] jacobian The Jacobian.
  virtual void plusJacobian(const double* /*unused: x*/,
                            double* jacobian) const {
    Eigen::Map<
        Eigen::Matrix<double, nTimeMinDim, nTimeMinDim, Eigen::RowMajor> >
        identity(jacobian);
    identity.setIdentity();
  }

  // Delta_Chi=x0_plus_Delta[-]x0
  /// \brief Computes the minimal difference between a variable x and a
  /// perturbed variable x_plus_delta
  /// @param[in] x0 Variable.
  /// @param[in] x0_plus_Delta Perturbed variable.
  /// @param[out] Delta_Chi Minimal difference.
  /// \return True on success.
  virtual void minus(const double* x0, const double* x0_plus_Delta,
                     double* Delta_Chi) const {
    Eigen::Map<const Eigen::Matrix<double, nTimeDim, 1> > x0_(x0);
    Eigen::Map<Eigen::Matrix<double, nTimeDim, 1> > Delta_Chi_(Delta_Chi);
    Eigen::Map<const Eigen::Matrix<double, nTimeDim, 1> > x0_plus_Delta_(
        x0_plus_Delta);
    Delta_Chi_ = x0_plus_Delta_ - x0_;
  }

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  //  /// @param[in] x0 Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  virtual void liftJacobian(const double* /*unused: x*/,
                            double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, nTimeMinDim, nTimeDim, Eigen::RowMajor> >
        identity(jacobian);
    identity.setIdentity();
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "CameraTimeParamBlock"; }

 private:
  okvis::Time timestamp_;  ///< Time of this state.
};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_CAMERATIMEPARAMBLOCK_HPP_ */
