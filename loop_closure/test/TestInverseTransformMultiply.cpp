#include <gtest/gtest.h>
#include <okvis/InverseTransformMultiplyJacobian.hpp>
#include <okvis/kinematics/sophus_operators.hpp>

class InverseTransformMultiplyTest : public ::testing::Test {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 protected:

  void SetUp() override {
    srand((unsigned int)time(0));  // comment this for deterministic behavior
    T_WA_.setRandom();
    T_WB_.setRandom();
    tmj_.initialize(T_WA_, T_WB_);
    T_AB_ = tmj_.multiplyT();
    computeNumericJacobians();
  }

  void computeNumericJacobians() {
    Eigen::Matrix<double, 6, 1> delta;

    for (int i = 0; i < 6; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_WA_bar = T_WA_;
      T_WA_bar.oplus(delta);
      okvis::InverseTransformMultiplyJacobian tmj_bar(T_WA_bar, T_WB_);
      okvis::kinematics::Transformation T_AB_bar = tmj_bar.multiplyT();
      Eigen::Matrix<double, 6, 1> ratio =
          okvis::kinematics::ominus(T_AB_bar, T_AB_) / eps;
      dp_ddelta_WA_.col(i) = ratio.head<3>();
      dtheta_ddelta_WA_.col(i) = ratio.tail<3>();
    }

    for (int i = 0; i < 6; ++i) {
      delta.setZero();
      delta(i) = eps;
      okvis::kinematics::Transformation T_WB_bar = T_WB_;
      T_WB_bar.oplus(delta);
      okvis::InverseTransformMultiplyJacobian tmj_bar(T_WA_, T_WB_bar);
      okvis::kinematics::Transformation T_AB_bar = tmj_bar.multiplyT();
      Eigen::Matrix<double, 6, 1> ratio =
          okvis::kinematics::ominus(T_AB_bar, T_AB_) / eps;
      dp_ddelta_WB_.col(i) = ratio.head<3>();
      dtheta_ddelta_WB_.col(i) = ratio.tail<3>();
    }
  }

  // void TearDown() override {}
  okvis::kinematics::Transformation T_WA_;
  okvis::kinematics::Transformation T_WB_;
  okvis::InverseTransformMultiplyJacobian tmj_;
  okvis::kinematics::Transformation T_AB_;

  Eigen::Matrix<double, 3, 6> dtheta_ddelta_WA_;
  Eigen::Matrix<double, 3, 6> dp_ddelta_WA_;
  Eigen::Matrix<double, 3, 6> dtheta_ddelta_WB_;
  Eigen::Matrix<double, 3, 6> dp_ddelta_WB_;

  const double eps = 1e-6;
};

TEST_F(InverseTransformMultiplyTest, translationJacobians) {
  Eigen::Matrix3d dp_dtheta_WA, dtheta_dtheta_WB, dp_dt_WA, dp_dt_WB;
  tmj_.dp_dp_WA(&dp_dt_WA);
  tmj_.dp_dp_WB(&dp_dt_WB);
  tmj_.dp_dtheta_WA(&dp_dtheta_WA);
  EXPECT_LT((dp_dt_WA - dp_ddelta_WA_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dt_WA\n" << dp_dt_WA << "\ndp_ddelta_WA_\n" << dp_ddelta_WA_.topLeftCorner<3, 3>();
  EXPECT_LT((dp_dtheta_WA - dp_ddelta_WA_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "\ndp_ddelta_WA.topRightCorner<3, 3>()\n" << dp_ddelta_WA_.topRightCorner<3, 3>();
  EXPECT_LT((dp_dt_WB - dp_ddelta_WB_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dp_dt_WB\n" << dp_dt_WB << "\ndp_ddelta_WB_.topLeftCorner<3, 3>()\n"
                 << dp_ddelta_WB_.topLeftCorner<3, 3>();
  EXPECT_LT((dp_ddelta_WB_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
}

TEST_F(InverseTransformMultiplyTest, rotationJacobians) {
  Eigen::Matrix3d dtheta_dtheta_WA, dtheta_dtheta_WB, dtheta_dp_WA,
      dtheta_dp_WB;
  tmj_.dtheta_dtheta_WA(&dtheta_dtheta_WA);
  tmj_.dtheta_dtheta_WB(&dtheta_dtheta_WB);

  EXPECT_LT((dtheta_ddelta_WA_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((dtheta_dtheta_WA - dtheta_ddelta_WA_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dtheta_dtheta_WA\n" << dtheta_dtheta_WA << "\ndtheta_ddelta_WA_.topRightCorner<3, 3>()\n"
                 << dtheta_ddelta_WA_.topRightCorner<3, 3>();
  EXPECT_LT((dtheta_ddelta_WB_.topLeftCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((dtheta_dtheta_WB - dtheta_ddelta_WB_.topRightCorner<3, 3>())
                .lpNorm<Eigen::Infinity>(),
            eps) << "dtheta_dtheta_WB\n" << dtheta_dtheta_WB << "\n dtheta_ddelta_WB_.topRightCorner<3, 3>()\n"
                 << dtheta_ddelta_WB_.topRightCorner<3, 3>();
}

TEST_F(InverseTransformMultiplyTest, bundledJacobians) {
  Eigen::Matrix<double, 6, 6> Jzx, Jzy;
  tmj_.dT_dT_WA(&Jzx);
  tmj_.dT_dT_WB(&Jzy);
  EXPECT_LT((Jzx.topRows<3>() - dp_ddelta_WA_).lpNorm<Eigen::Infinity>(), eps);
  EXPECT_LT((Jzx.bottomRows<3>() - dtheta_ddelta_WA_).lpNorm<Eigen::Infinity>(),
            eps);
  EXPECT_LT((Jzy.topRows<3>() - dp_ddelta_WB_).lpNorm<Eigen::Infinity>(), eps);
  EXPECT_LT((Jzy.bottomRows<3>() - dtheta_ddelta_WB_).lpNorm<Eigen::Infinity>(),
            eps);
}
