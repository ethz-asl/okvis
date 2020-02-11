#include <gtest/gtest.h>
#include <glog/logging.h>

#include <msckf/VectorNormalizationJacobian.hpp>
#include <okvis/ceres/MarginalizationError.hpp>
#include <Eigen/Core>

TEST(MarginalizationError, pseudoInverseSymmSqrtFullRank) {
  double rootScale = 5;
  double scale = rootScale * rootScale;
  Eigen::Matrix3d lambda = Eigen::Matrix3d::Identity() * scale;
  Eigen::Matrix3d invLambda = Eigen::Matrix3d::Identity() / scale;
  Eigen::Matrix3d invLambdaSqrt = Eigen::Matrix3d::Identity() / rootScale;

  int rankExpected = 3;
  Eigen::Matrix3d result;
  int rank;
  okvis::ceres::MarginalizationError::pseudoInverseSymm(
      lambda, result, std::numeric_limits<double>::epsilon(), &rank);
  EXPECT_EQ(rank, rankExpected);
  const double eps = 1e-7;
  EXPECT_TRUE(invLambda.isApprox(result, eps));

  Eigen::Matrix3d resultSqrt;
  int rankSqrt;
  okvis::ceres::MarginalizationError::pseudoInverseSymmSqrt(
      lambda, resultSqrt, std::numeric_limits<double>::epsilon(), &rankSqrt);

  EXPECT_EQ(rankSqrt, rankExpected);
  EXPECT_TRUE(invLambdaSqrt.isApprox(resultSqrt, eps));
}

class SimpleProjection {
 public:
  SimpleProjection () {
    K << 600, 0, 320, 0, 600, 240, 0, 0, 1;
    Kinv << 1.0 / 600, 0, -320.0 / 600, 0, 1.0 / 600, -240.0 / 600, 0, 0, 1;
  }
  /**
   * @brief projectWithJacobian
   * @param xy1
   * @param uv
   * @param j d(u,v)/d(x,y)
   */
  void projectWithJacobian(const Eigen::Vector3d& xy1, Eigen::Vector2d& uv,
                           Eigen::Matrix<double, 2, 2>* j) const {
    uv = (K * xy1).head<2>();
    *j = K.topLeftCorner<2, 2>();
  }

  void backProjectWithJacobian(const Eigen::Vector2d& uv, Eigen::Vector3d& xy1,
                               Eigen::Matrix<double, 2, 2>* j) const {
    Eigen::Vector3d uv1;
    uv1 << uv, 1;
    xy1 = Kinv * uv1;
    *j = Kinv.topLeftCorner<2, 2>();
  }

  Eigen::Matrix3d K;
  Eigen::Matrix3d Kinv;
};

TEST(MarginalizationError, pseudoInverseSymmSqrt) {
  Eigen::Vector2d uv = Eigen::Vector2d::Random();
  uv[0] *= 320;
  uv[1] *= 240;

  Eigen::Vector3d xy1;
  SimpleProjection camera;
  Eigen::Matrix2d dxy_duv;
  camera.backProjectWithJacobian(uv, xy1, &dxy_duv);
  Eigen::Matrix<double, 3, 2> dxy1_duv;
  dxy1_duv.setZero();
  dxy1_duv.topLeftCorner<2, 2>() = dxy_duv;

  msckf::VectorNormalizationJacobian normalJac(xy1);
  Eigen::Matrix3d dunit_dxy1;
  normalJac.dxi_dvec(&dunit_dxy1);
  Eigen::Vector2d covuvDiagonal = Eigen::Vector2d::Random();
  covuvDiagonal[0] += 0.5;
  covuvDiagonal[1] += 0.5;
  Eigen::Matrix2d covuv = covuvDiagonal.asDiagonal();
  Eigen::Matrix2d infouv = Eigen::Matrix2d::Identity();
  infouv(0, 0) /= covuvDiagonal[0];
  infouv(1, 1) /= covuvDiagonal[1];

  Eigen::Matrix<double, 3, 2> dunit_duv = dunit_dxy1 * dxy1_duv;
  Eigen::Matrix3d covxi = dunit_duv * covuv * dunit_duv.transpose();
  int rankExpected = 2;
  Eigen::Matrix3d blindMatrix = Eigen::Matrix3d::Identity();
  blindMatrix(2, 2) = 0;

  Eigen::Matrix3d result;
  int rank;
  okvis::ceres::MarginalizationError::pseudoInverseSymm(
      covxi, result, std::numeric_limits<double>::epsilon(), &rank);
  EXPECT_EQ(rank, rankExpected);
  Eigen::Matrix3d dev = result * covxi - blindMatrix;
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(), 1.0);
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(),
            1e-6 * result.lpNorm<Eigen::Infinity>())
      << "covxi\n"
      << covxi << "\nresult\n"
      << result;
  Eigen::Matrix3d resultSqrt;
  int rankSqrt;
  okvis::ceres::MarginalizationError::pseudoInverseSymmSqrt(
      covxi, resultSqrt, std::numeric_limits<double>::epsilon(), &rankSqrt);
  EXPECT_EQ(rankSqrt, rankExpected);
  dev = resultSqrt * resultSqrt.transpose() * covxi - blindMatrix;
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(), 1.0);
  EXPECT_LT(dev.lpNorm<Eigen::Infinity>(),
            1e-3 * resultSqrt.lpNorm<Eigen::Infinity>())
      << "covxi\n"
      << covxi << "\nresultSqrt\n"
      << resultSqrt;

  Eigen::MatrixXd pinv =
      covxi.completeOrthogonalDecomposition().pseudoInverse();
  EXPECT_LT((pinv - result).lpNorm<Eigen::Infinity>(), 1e-8)
      << "eigen pinv\n"
      << pinv << "\nokvis pinv\n"
      << result;
}
