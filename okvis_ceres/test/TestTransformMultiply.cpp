#include <gtest/gtest.h>

#include <Eigen/Core>
#include <msckf/JacobianHelpers.hpp>
#include <msckf/TransformMultiplyJacobian.hpp>
#include <okvis/kinematics/Transformation.hpp>

class TransformMultiplyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    srand((unsigned int)time(0));  // comment this for deterministic behavior
    T_AB_.setRandom();
    T_BC_.setRandom();
    tmj_.initialize(T_AB_, T_BC_);
    T_AC_ = tmj_.multiplyT();
    computeNumericJacobians();
  }

  void computeNumericJacobians() {
    Eigen::Matrix<double, 6, 1> delta;

    for (int i = 0; i < 6; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_AB_bar = T_AB_;
      T_AB_bar.oplus(delta);
      msckf::TransformMultiplyJacobian tmj_bar(T_AB_bar, T_BC_);
      okvis::kinematics::Transformation T_AC_bar = tmj_bar.multiplyT();
      Eigen::Matrix<double, 6, 1> ratio =
          okvis::ceres::ominus(T_AC_bar, T_AC_) / eps;
      dp_ddelta_AB_.col(i) = ratio.head<3>();
      dtheta_ddelta_AB_.col(i) = ratio.tail<3>();
    }

    for (int i = 0; i < 6; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_BC_bar = T_BC_;
      T_BC_bar.oplus(delta);
      msckf::TransformMultiplyJacobian tmj_bar(T_AB_, T_BC_bar);
      okvis::kinematics::Transformation T_AC_bar = tmj_bar.multiplyT();
      Eigen::Matrix<double, 6, 1> ratio =
          okvis::ceres::ominus(T_AC_bar, T_AC_) / eps;
      dp_ddelta_BC_.col(i) = ratio.head<3>();
      dtheta_ddelta_BC_.col(i) = ratio.tail<3>();
    }
  }

  // void TearDown() override {}
  okvis::kinematics::Transformation T_AB_;
  okvis::kinematics::Transformation T_BC_;
  msckf::TransformMultiplyJacobian tmj_;
  okvis::kinematics::Transformation T_AC_;

  Eigen::Matrix<double, 3, 6> dtheta_ddelta_AB_;
  Eigen::Matrix<double, 3, 6> dp_ddelta_AB_;
  Eigen::Matrix<double, 3, 6> dtheta_ddelta_BC_;
  Eigen::Matrix<double, 3, 6> dp_ddelta_BC_;

  const double eps = 1e-6;
};

TEST_F(TransformMultiplyTest, translationJacobians) {
  Eigen::Matrix3d dp_dtheta_AB, dp_dtheta_BC, dp_dt_AB, dp_dt_BC;
  tmj_.dp_dtheta_AB(&dp_dtheta_AB);
  tmj_.dp_dp_AB(&dp_dt_AB);
  tmj_.dp_dtheta_BC(&dp_dtheta_BC);
  tmj_.dp_dp_BC(&dp_dt_BC);
  EXPECT_LT((dp_dt_AB - dp_ddelta_AB_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dt_AB\n" << dp_dt_AB << "\ndp_ddelta_AB_\n" << dp_ddelta_AB_.topLeftCorner<3, 3>();
  EXPECT_LT((dp_dtheta_AB - dp_ddelta_AB_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dtheta_AB\n" << dp_dtheta_AB <<
                    "\ndp_ddelta_AB_.topRightCorner<3, 3>()\n" << dp_ddelta_AB_.topRightCorner<3, 3>();
  EXPECT_LT((dp_dt_BC - dp_ddelta_BC_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dt_BC\n" << dp_dt_BC << "\ndp_ddelta_BC_.topLeftCorner<3, 3>()\n"
                 << dp_ddelta_BC_.topLeftCorner<3, 3>();
  EXPECT_LT((dp_dtheta_BC - dp_ddelta_BC_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
}

TEST_F(TransformMultiplyTest, rotationJacobians) {
  Eigen::Matrix3d dtheta_dtheta_AB, dtheta_dtheta_BC, dtheta_dp_AB,
      dtheta_dp_BC;
  tmj_.dtheta_dtheta_AB(&dtheta_dtheta_AB);
  tmj_.dtheta_dp_AB(&dtheta_dp_AB);
  tmj_.dtheta_dtheta_BC(&dtheta_dtheta_BC);
  tmj_.dtheta_dp_BC(&dtheta_dp_BC);
  EXPECT_LT((dtheta_dp_AB - dtheta_ddelta_AB_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((dtheta_dtheta_AB - dtheta_ddelta_AB_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dtheta_dtheta_AB\n" << dtheta_dtheta_AB << "\ndtheta_ddelta_AB_.topRightCorner<3, 3>()\n"
                 << dtheta_ddelta_AB_.topRightCorner<3, 3>();
  EXPECT_LT((dtheta_dp_BC - dtheta_ddelta_BC_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((dtheta_dtheta_BC - dtheta_ddelta_BC_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dtheta_dtheta_BC\n" << dtheta_dtheta_BC << "\n dtheta_ddelta_BC_.topRightCorner<3, 3>()\n"
                 << dtheta_ddelta_BC_.topRightCorner<3, 3>();
}
