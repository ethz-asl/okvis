#include <gtest/gtest.h>
#include <okvis/cameras/FovDistortion.hpp>

TEST(FovDistortion, UndistortAndJacobian) {
  using namespace okvis;
  using namespace okvis::cameras;
  FovDistortion d(0.6);
  Eigen::MatrixXd Jd, estJd;

  Eigen::Vector2d x0(3.0E-1, 3.0E-1);
  Eigen::Vector2d x1(3.0E-1, 3.0E-1);
  Eigen::Vector2d xtemp;
  d.distort(x1, &xtemp);
  d.undistort(xtemp, &x1);

  double pointDiff = (x0 - x1).lpNorm<Eigen::Infinity>();
  EXPECT_LT(pointDiff, 1e-8);

  Eigen::Matrix2d pointJacobian, pointJacobianExpected;
  d.distort(x0, &xtemp, &pointJacobian, NULL);

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

  double jacDiffNorm =
      (pointJacobianExpected - pointJacobian).lpNorm<Eigen::Infinity>();

  EXPECT_LT(jacDiffNorm, 1.0e-5);
}
