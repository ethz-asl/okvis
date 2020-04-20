#ifndef INCLUDE_OKVIS_MATRIX_PSEUDO_INVERSE_HPP
#define INCLUDE_OKVIS_MATRIX_PSEUDO_INVERSE_HPP

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <okvis/assert_macros.hpp>

namespace okvis {
class MatrixPseudoInverse
{
public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)
  /**
   * @brief Pseudo inversion of a symmetric matrix.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @param[in] a Input Matrix
   * @param[out] result Output, i.e. pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @param[out] rank Optional rank.
   * @return
   */
  template<typename Derived>
  static bool pseudoInverseSymm(
      const Eigen::MatrixBase<Derived>&a,
      const Eigen::MatrixBase<Derived>&result, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon(), int * rank = 0);

  /**
   * @brief Pseudo inversion and square root (Cholesky decomposition) of a symmetric matrix.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @param[in] a Input Matrix
   * @param[out] result Output, i.e. the Cholesky decomposition of a pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @param[out] rank The rank, if of interest.
   * @return
   */
  template<typename Derived>
  static bool pseudoInverseSymmSqrt(
      const Eigen::MatrixBase<Derived>&a,
      const Eigen::MatrixBase<Derived>&result, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon(),
      int* rank = NULL);

  /**
   * @brief Block-wise pseudo inversion of a symmetric matrix with non-zero diagonal blocks.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @tparam blockDim The block size of the diagonal blocks.
   * @param[in] M_in Input Matrix
   * @param[out] M_out Output, i.e. thepseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @return
   */
  template<typename Derived, int blockDim>
  static void blockPinverse(
      const Eigen::MatrixBase<Derived>& M_in,
      const Eigen::MatrixBase<Derived>& M_out, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon());


  /**
   * @brief Block-wise pseudo inversion and square root (Cholesky decomposition)
   *        of a symmetric matrix with non-zero diagonal blocks.
   * @warning   This uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
   *            (negative Eigenvalues are set to zero).
   * @tparam Derived Matrix type (auto-deducible).
   * @tparam blockDim The block size of the diagonal blocks.
   * @param[in] M_in Input Matrix
   * @param[out] M_out Output, i.e. the Cholesky decomposition of a pseudo-inverse.
   * @param[in] epsilon The tolerance.
   * @return
   */
  template<typename Derived, int blockDim>
  static void blockPinverseSqrt(
      const Eigen::MatrixBase<Derived>& M_in,
      const Eigen::MatrixBase<Derived>& M_out, double epsilon =
          std::numeric_limits<typename Derived::Scalar>::epsilon());

};

// Pseudo inversion of a symmetric matrix.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
// (negative Eigenvalues are set to zero)
template<typename Derived>
bool MatrixPseudoInverse::pseudoInverseSymm(
    const Eigen::MatrixBase<Derived>&a, const Eigen::MatrixBase<Derived>&result,
    double epsilon, int * rank) {

  OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                        "matrix supplied is not quadratic");

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols()
      * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
      * Eigen::VectorXd(
          (saes.eigenvalues().array() > tolerance).select(
              saes.eigenvalues().array().inverse(), 0)).asDiagonal()
      * (saes.eigenvectors().transpose());

  if (rank) {
    *rank = 0;
    for (int i = 0; i < a.rows(); ++i) {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }

  return true;
}

// Pseudo inversion and square root (Cholesky decomposition) of a symmetric matrix.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
// (negative Eigenvalues are set to zero)
template<typename Derived>
bool MatrixPseudoInverse::pseudoInverseSymmSqrt(
    const Eigen::MatrixBase<Derived>&a, const Eigen::MatrixBase<Derived>&result,
    double epsilon, int * rank) {

  OKVIS_ASSERT_TRUE_DBG(Exception, a.rows() == a.cols(),
                        "matrix supplied is not quadratic");

  Eigen::SelfAdjointEigenSolver<Derived> saes(a);

  typename Derived::Scalar tolerance = epsilon * a.cols()
      * saes.eigenvalues().array().maxCoeff();

  const_cast<Eigen::MatrixBase<Derived>&>(result) = (saes.eigenvectors())
      * Eigen::VectorXd(
          Eigen::VectorXd(
              (saes.eigenvalues().array() > tolerance).select(
                  saes.eigenvalues().array().inverse(), 0)).array().sqrt())
          .asDiagonal();

  if (rank) {
    *rank = 0;
    for (int i = 0; i < a.rows(); ++i) {
      if (saes.eigenvalues()[i] > tolerance)
        (*rank)++;
    }
  }

  return true;
}

// Block-wise pseudo inversion of a symmetric matrix with non-zero diagonal blocks.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
// (negative Eigenvalues are set to zero)
template<typename Derived, int blockDim>
void MatrixPseudoInverse::blockPinverse(
    const Eigen::MatrixBase<Derived>& M_in,
    const Eigen::MatrixBase<Derived>& M_out, double epsilon) {

  OKVIS_ASSERT_TRUE_DBG(Exception, M_in.rows() == M_in.cols(),
                        "matrix supplied is not quadratic");

  const_cast<Eigen::MatrixBase<Derived>&>(M_out).resize(M_in.rows(),
                                                        M_in.rows());
  const_cast<Eigen::MatrixBase<Derived>&>(M_out).setZero();
  for (int i = 0; i < M_in.cols(); i += blockDim) {
    Eigen::Matrix<double, blockDim, blockDim> inv;
    const Eigen::Matrix<double, blockDim, blockDim> in = M_in
        .template block<blockDim, blockDim>(i, i);
    //const Eigen::Matrix<double,blockDim,blockDim> in1=0.5*(in+in.transpose());
    pseudoInverseSymm(in, inv, epsilon);
    const_cast<Eigen::MatrixBase<Derived>&>(M_out)
        .template block<blockDim, blockDim>(i, i) = inv;
  }
}

// Block-wise pseudo inversion and square root (Cholesky decomposition)
// of a symmetric matrix with non-zero diagonal blocks.
// attention: this uses Eigen-decomposition, it assumes the input is symmetric positive semi-definite
// (negative Eigenvalues are set to zero)
template<typename Derived, int blockDim>
void MatrixPseudoInverse::blockPinverseSqrt(
    const Eigen::MatrixBase<Derived>& M_in,
    const Eigen::MatrixBase<Derived>& M_out, double epsilon) {

  OKVIS_ASSERT_TRUE_DBG(Exception, M_in.rows() == M_in.cols(),
                        "matrix supplied is not quadratic");

  const_cast<Eigen::MatrixBase<Derived>&>(M_out).resize(M_in.rows(),
                                                        M_in.rows());
  const_cast<Eigen::MatrixBase<Derived>&>(M_out).setZero();
  for (int i = 0; i < M_in.cols(); i += blockDim) {
    Eigen::Matrix<double, blockDim, blockDim> inv;
    const Eigen::Matrix<double, blockDim, blockDim> in = M_in
        .template block<blockDim, blockDim>(i, i);
    //const Eigen::Matrix<double,blockDim,blockDim> in1=0.5*(in+in.transpose());
    pseudoInverseSymmSqrt(in, inv, epsilon);
    const_cast<Eigen::MatrixBase<Derived>&>(M_out)
        .template block<blockDim, blockDim>(i, i) = inv;
  }
}

} // namespace okvis
#endif // INCLUDE_OKVIS_MATRIX_PSEUDO_INVERSE_HPP
