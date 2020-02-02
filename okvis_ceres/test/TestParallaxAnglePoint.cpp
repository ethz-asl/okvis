#include <gtest/gtest.h>
#include <msckf/ParallaxAnglePoint.hpp>

class ParallaxAnglePointTest : public ::testing::Test {
 protected:
  ParallaxAnglePointTest() {}
  void SetUp() override {
    s_ = 47;
    pap_.setRandom(s_);
    quatAndCS_.setRandom();
    quatAndCS_[5] = std::fabs(quatAndCS_[5]);
    double norm = std::sqrt(quatAndCS_[4] * quatAndCS_[4] + quatAndCS_[5] * quatAndCS_[5]);
    quatAndCS_[4] /= norm;
    quatAndCS_[5] /= norm;
    vecAndTheta_.setRandom();
    vecAndTheta_[3] = std::fabs(vecAndTheta_[3]);
    pap_ = LWF::ParallaxAnglePoint(vecAndTheta_.head<3>(), std::cos(vecAndTheta_[3]));
  }

  LWF::ParallaxAnglePoint pap_;
  Eigen::Matrix<double, 4, 1> vecAndTheta_;
  Eigen::Matrix<double, 6, 1> quatAndCS_;
  unsigned int s_;
  const double eps = 1e-8;
};

TEST_F(ParallaxAnglePointTest, CopyAndSet) {
  std::vector<double> params;
  pap_.set(quatAndCS_.data());
  pap_.copy(&params);
  Eigen::Map<const Eigen::Matrix<double, 6, 1>> paramVector(params.data());
  EXPECT_LT((paramVector - quatAndCS_).lpNorm<Eigen::Infinity>(), eps);
}

TEST_F(ParallaxAnglePointTest, GetBearing) {
  EXPECT_LT((pap_.getVec() - vecAndTheta_.head<3>().normalized())
                .lpNorm<Eigen::Infinity>(),
            eps);
}

TEST_F(ParallaxAnglePointTest, GetTheta) {
  EXPECT_LT(std::fabs(pap_.getAngle() - vecAndTheta_[3]), eps);
}



class AngleElementTest : public ::testing::Test {
 protected:
  AngleElementTest() {}
  void SetUp() override {
    theta_ = Eigen::Matrix<double, 2, 1>::Random();
    theta_ = theta_.cwiseAbs();
    ct_ << std::cos(theta_[0]), std::cos(theta_[1]);
    angle0_ = LWF::AngleElement(ct_[0]);
    angle1_ = LWF::AngleElement(ct_[1]);
  }

  Eigen::Matrix<double, 2, 1> theta_;
  Eigen::Matrix<double, 2, 1> ct_;
  LWF::AngleElement angle0_;
  LWF::AngleElement angle1_;
};

TEST_F(AngleElementTest, assignOperator) {
  LWF::AngleElement angle2(ct_[1]);
  LWF::AngleElement angle3 = angle2;
  EXPECT_NEAR(angle2.getAngle(), theta_[1], 1e-7);
  EXPECT_NEAR(angle1_.getAngle(), theta_[1], 1e-7);
  EXPECT_NEAR(angle3.getAngle(), theta_[1], 1e-7);
}

TEST_F(AngleElementTest, boxMinus) {
  Eigen::Matrix<double, 1, 1> delta;
  angle1_.boxMinus(angle0_, delta);
  EXPECT_NEAR(delta[0], theta_[1] - theta_[0], 1e-7);
}

TEST_F(AngleElementTest, boxMinus2) {
  Eigen::Matrix<double, 1, 1> delta;
  angle1_.boxMinus(angle0_, delta);
  LWF::AngleElement angle2 = angle0_;
  angle2.boxPlus(delta, angle2);
  EXPECT_NEAR(angle2.getAngle(), angle1_.getAngle(), 1e-7);
}

TEST_F(AngleElementTest, boxMinusJac) {
  Eigen::Matrix<double, 1, 1> delta;
  angle1_.boxMinus(angle0_, delta);
  Eigen::MatrixXd jac;
  angle1_.boxMinusJac(angle0_, jac);
  double h = 0.00001;
  LWF::AngleElement angle2(std::cos(theta_[1] + h));
  Eigen::Matrix<double, 1, 1> deltah;
  angle2.boxMinus(angle0_, deltah);

  EXPECT_NEAR(jac(0, 0), (deltah(0, 0) - delta(0, 0)) / h, 1e-7);
}
