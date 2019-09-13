#include <gtest/gtest.h>
#include <okvis/cameras/FovDistortion.hpp>

void undistortAndJacobian(const Eigen::Vector2d& undistorted, const double omega) {
  using namespace okvis;
  using namespace okvis::cameras;
  FovDistortion d(omega);
  Eigen::Vector2d x0 = undistorted;
  Eigen::Vector2d x1 = undistorted;
  Eigen::Vector2d xtemp;
  d.distort(x1, &xtemp);
  d.undistort(xtemp, &x1);

  double pointDiff = (x0 - x1).lpNorm<Eigen::Infinity>();
  EXPECT_LT(pointDiff, 1e-8);

  Eigen::Matrix2d pointJacobian, pointJacobianExpected;
  Eigen::Matrix2Xd distortJacobian, distortJacobianExpected;
  d.distort(x0, &xtemp, &pointJacobian, &distortJacobian);

  d.distort(x0, &xtemp);
  Eigen::Vector2d xtemp1;
  double step = 1e-5;
  x1 = x0;
  x1[0] += step;
  d.distort(x1, &xtemp1);
  pointJacobianExpected.topLeftCorner<2, 1>() = (xtemp1 - xtemp) / step;

  x1 = x0;
  x1[1] += step;
  d.distort(x1, &xtemp1);
  pointJacobianExpected.topRightCorner<2, 1>() = (xtemp1 - xtemp) / step;

  Eigen::VectorXd distortParams;
  d.getParameters(distortParams);
  FovDistortion d1(distortParams[0] + step);
  d1.distort(x0, &xtemp1);
  distortJacobianExpected = (xtemp1 - xtemp) / step;

  double jacDiffNorm =
      (pointJacobianExpected - pointJacobian).lpNorm<Eigen::Infinity>();
  EXPECT_LT(jacDiffNorm, 1.0e-5);

  double distortJacDiffNorm = (distortJacobianExpected - distortJacobian).lpNorm<Eigen::Infinity>();
  std::cout << "analytic distort Jac\n" << distortJacobian
            << "\nnumeric Jac\n" << distortJacobianExpected << "\n";
  EXPECT_LT(distortJacDiffNorm, 1.0e-5);
}

TEST(FovDistortion, UndistortAndJacobian) {
    undistortAndJacobian(Eigen::Vector2d(3.0e-1, 2.0e-1), 0.9);
}

TEST(FovDistortion, UndistortAndJacobianSmallOmega) {
    undistortAndJacobian(Eigen::Vector2d(3.0e-1, 3.0e-1), 1e-8);
}

TEST(FovDistortion, UndistortAndJacobianSmallRadius) {
    undistortAndJacobian(Eigen::Vector2d(1.0e-8, 1.0e-8), 0.8);
}

