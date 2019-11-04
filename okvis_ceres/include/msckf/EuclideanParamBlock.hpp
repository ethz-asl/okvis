
/**
 * @file EuclideanParamBlock.hpp
 * @brief Header file for the EuclideanParamBlock class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCK_HPP_
#define INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCK_HPP_

#include <okvis/ceres/ParameterBlockDynamic.hpp>
#include <Eigen/Core>
#include <okvis/Time.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Wraps the parameter block for camera projection intrinsic estimate
class EuclideanParamBlock
    : public ParameterBlockDynamic {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> estimate_t;
  /// \brief Set the time.
  /// @param[in] timestamp The timestamp of this state.
  void setTimestamp(const okvis::Time& timestamp) { timestamp_ = timestamp; }

  /// \brief Default constructor (assumes not fixed).
  EuclideanParamBlock() : ParameterBlockDynamic(0, 0) {
    ParameterBlock::setFixed(false);
  }

  /// \brief Trivial destructor.
  virtual ~EuclideanParamBlock() {
  }

  /// \brief Constructor with estimate and time.
  /// @param[in] ProjectionIntrinsicParams
  /// @param[in] id The (unique) ID of this block.
  /// @param[in] timestamp The timestamp of this state.
  /// @param[in] dim The dim and minDim of this state.
  EuclideanParamBlock(const Eigen::VectorXd& params, uint64_t id,
      const okvis::Time& timestamp, const int dim) :
      ParameterBlockDynamic(dim, dim) {
    setEstimate(params);
    setId(id);
    setTimestamp(timestamp);
    setFixed(false);
  }

  // setters
  /// @brief Set estimate of this parameter block.
  /// @param[in] ProjectionIntrinsicParams The estimate to set this to.
  void setEstimate(
      const Eigen::VectorXd& params) {
    for (int i = 0; i < dim_; ++i)
      parameters_[i] = params[i];
  }

  /// getters
  /// @brief Get estimate.
  Eigen::VectorXd estimate() const {
    Eigen::VectorXd params(dim_);
    for (int i = 0; i < dim_; ++i)
      params[i] = parameters_[i];
    return params;
  }

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
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > x0_(x0, dim_);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > Delta_Chi_(
        Delta_Chi, minDim_);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > x0_plus_Delta_(
        x0_plus_Delta, dim_);
    x0_plus_Delta_ = x0_ + Delta_Chi_;
  }

  /// \brief The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  /// @param[in] x0 Variable.
  /// @param[out] jacobian The Jacobian.
  virtual void plusJacobian(const double* /*unused: x*/,
                            double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor> >
        identity(jacobian, dim_, minDim_);
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
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > x0_(x0, dim_);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1> > Delta_Chi_(Delta_Chi, minDim_);
    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 1> > x0_plus_Delta_(
        x0_plus_Delta, dim_);
    Delta_Chi_ = x0_plus_Delta_ - x0_;
  }

  /// \brief Computes the Jacobian from minimal space to naively
  /// overparameterised space as used by ceres.
  //  /// @param[in] x0 Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  virtual void liftJacobian(const double* /*unused: x*/,
                            double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor> >
        identity(jacobian, minDim_, dim_);
    identity.setIdentity();
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "EuclideanParamBlock"; }

 private:
  okvis::Time timestamp_;  ///< Time of this state.
};
}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_EUCLIDEANPARAMBLOCK_HPP_ */
