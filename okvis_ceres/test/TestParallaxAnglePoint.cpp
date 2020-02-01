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
