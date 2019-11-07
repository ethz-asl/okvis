
/**
 * @file EuclideanParamErorr.hpp
 * @brief Source file for the EuclideanParamErorr class.
 * @author Jianzhu Huai
 */

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Construct with measurement and information matrix
template<int kParamDim>
EuclideanParamError<kParamDim>::EuclideanParamError(const Eigen::Matrix<double, kParamDim, 1> & measurement,
                                     const information_t & information) {
  setMeasurement(measurement);
  setInformation(information);
}

// Construct with measurement and variance.
template<int kParamDim>
EuclideanParamError<kParamDim>::EuclideanParamError(const Eigen::Matrix<double, kParamDim, 1>& measurement,
                                     const Eigen::Matrix<double, kParamDim, 1>& variance) {
  setMeasurement(measurement);

  information_t information;
  information.setIdentity();
  information.diagonal().cwiseQuotient(variance);
  setInformation(information);
}

// Set the information.
template<int kParamDim>
void EuclideanParamError<kParamDim>::setInformation(const information_t & information) {
  information_ = information;
  covariance_ = information.inverse();
  // perform the Cholesky decomposition on order to obtain the correct error weighting
  Eigen::LLT<information_t> lltOfInformation(information_);
  squareRootInformation_ = lltOfInformation.matrixL().transpose();
}

// This evaluates the error term and additionally computes the Jacobians.
template<int kParamDim>
bool EuclideanParamError<kParamDim>::Evaluate(double const* const * parameters,
                                 double* residuals, double** jacobians) const {
  return EvaluateWithMinimalJacobians(parameters, residuals, jacobians, NULL);
}

// This evaluates the error term and additionally computes
// the Jacobians in the minimal internal representation.
template<int kParamDim>
bool EuclideanParamError<kParamDim>::EvaluateWithMinimalJacobians(
    double const* const * parameters, double* residuals, double** jacobians,
    double** jacobiansMinimal) const {

  // compute error
  Eigen::Map<const Eigen::Matrix<double, kParamDim, 1>> estimate(parameters[0]);
  Eigen::Matrix<double, kParamDim, 1> error = measurement_ - estimate;

  // weigh it
  Eigen::Map<Eigen::Matrix<double, kParamDim, 1> > weighted_error(residuals);
  weighted_error = squareRootInformation_ * error;

  // compute Jacobian - this is rather trivial in this case...
  if (jacobians != NULL) {
    if (jacobians[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, kParamDim, kParamDim, Eigen::RowMajor> > J0(
          jacobians[0]);
      J0 = -squareRootInformation_ * Eigen::Matrix<double, kParamDim, kParamDim>::Identity();
    }
  }
  if (jacobiansMinimal != NULL) {
    if (jacobiansMinimal[0] != NULL) {
      Eigen::Map<Eigen::Matrix<double, kParamDim, kParamDim, Eigen::RowMajor> > J0min(
          jacobiansMinimal[0]);
      J0min = -squareRootInformation_ * Eigen::Matrix<double, kParamDim, kParamDim>::Identity();
    }
  }

  return true;
}

}  // namespace ceres
}  // namespace okvis
