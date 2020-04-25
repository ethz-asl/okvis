#ifndef INCLUDE_MSCKF_BEARING_RESIDUALS_HPP_
#define INCLUDE_MSCKF_BEARING_RESIDUALS_HPP_
#include <Eigen/Core>
#include <memory>

#include <ceres/loss_function.h>
#include <ceres/sized_cost_function.h>

#include <msckf/ceres/corrector.h>
#include <msckf/DirectionFromParallaxAngleJacobian.hpp>
#include <msckf/VectorNormalizationJacobian.hpp>

namespace msckf {
/**
 * @brief orderedIndicesMAJ
 * @param maj indices of main anchor, associate anchor, and remaining frames.
 */
void orderedIndicesMAJ(const std::vector<size_t>& anchorIndices,
                       int totalObservations, std::vector<int>* maj);

class SimplePointSharedData {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SimplePointSharedData() {}

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      unitBearingList;
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WC_list;
  std::vector<size_t>
      anchorIndices;  // Indices of main and associate anchors in the lists.
};

// e = \xi(N_{i,j}) - R_{WC(t_{i,j})} * f_{i, j} if m_i != j
// e = n_i - f_{i, m_i} otherwise
class BearingResiduals {
 public:
  typedef double Scalar;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = 6,
    NUM_LOCAL_PARAMETERS = 3,
  };
  /**
   * @brief BearingResiduals
   * @param pointDataPtr
   * @param huberEpsilon  0.01 ~ 2 sin(-.5\theta) ~ 5(pixels) : 500(focal length)
   */
  BearingResiduals(std::shared_ptr<const SimplePointSharedData> pointDataPtr,
                   double huberEpsilon)
      : pointDataPtr_(pointDataPtr),
        numResidualBlocks_(pointDataPtr->T_WC_list.size()),
        numResiduals_(pointDataPtr->T_WC_list.size() * 3),
        loss_function_(new ::ceres::HuberLoss(huberEpsilon)) {
    orderedIndicesMAJ(pointDataPtr->anchorIndices, numResidualBlocks_,
                      &majIndices_);
  }

  int NumResiduals() const { return numResiduals_; }

  void correct(Eigen::Vector3d* residual,
               Eigen::Matrix<double, 3, 3, Eigen::RowMajor>* jacobian) const {
    double squared_norm = residual->squaredNorm();
    double rho[3];
    loss_function_->Evaluate(squared_norm, rho);
    msckf::ceres::internal::Corrector correct(squared_norm, rho);
    // Correct the jacobians for the loss function.
    correct.CorrectJacobian(kResidualDim, kPapLocalDim, residual->data(),
                            jacobian->data());
    correct.CorrectResiduals(kResidualDim, residual->data());
  }

  /**
   * @brief operator ()
   * @param parameters
   * @param residuals
   * @param jacobian jacobian corresponds to the data of a COLUMN MAJOR Eigen
   * Matrix.
   * @return
   */
  bool operator()(const double* parameters, double* residuals,
                  double* jacobian) const;

 private:
  std::shared_ptr<const SimplePointSharedData> pointDataPtr_;
  int numResidualBlocks_;
  const int numResiduals_;  // Dim of the entire residual vector.
  std::vector<int> majIndices_;
  std::shared_ptr<::ceres::LossFunction> loss_function_;
  static const int kResidualDim = 3;
  static const int kPapLocalDim = 3;
};

}  // namespace msckf
#endif  // INCLUDE_MSCKF_BEARING_RESIDUALS_HPP_
