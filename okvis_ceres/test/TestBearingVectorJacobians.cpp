#include <gtest/gtest.h>

#include "SimulatedMotionForParallaxAngleTest.hpp"

TEST(VectorNormalizationJacobian, Evaluate) {
  Eigen::Vector3d vec;
  vec.setRandom();
  msckf::VectorNormalizationJacobian vnj(vec);
  Eigen::Matrix3d jac;
  vnj.dxi_dvec(&jac);
  Eigen::Vector3d vec_delta;
  Eigen::Vector3d delta;
  Eigen::Vector3d normalized_vec = vnj.normalized();
  Eigen::Matrix3d numericJac;
  double eps = 1e-6;
  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = eps;
    vec_delta = vec + delta;
    numericJac.col(i) = vec_delta.normalized() - normalized_vec;
  }
  numericJac /= eps;
  EXPECT_LT((numericJac - jac).lpNorm<Eigen::Infinity>(), eps);
}

#define DirectionFromParallaxAngleSubTest(METHOD, JACTYPE, EPS)              \
  TEST(DirectionFromParallaxAngle, METHOD) {                                 \
    for (int i = simul::MotionType::Sideways;                                \
         i <= simul::MotionType::AssociateObserver; i++) {                   \
      simul::MotionType mt = static_cast<simul::MotionType>(i);              \
      simul::SimulatedMotionForParallaxAngleTest simulatedMotion(mt, false); \
      JACTYPE numericJac;                                                    \
      simulatedMotion.METHOD(&numericJac);                                   \
      JACTYPE analyticJac;                                                   \
      simulatedMotion.dfpaj_.METHOD(&analyticJac);                           \
      EXPECT_LT((numericJac - analyticJac).lpNorm<Eigen::Infinity>(), EPS)   \
          << #METHOD << ", " << mt << ", numeric\n"                          \
          << numericJac << "\nanalytic\n"                                    \
          << analyticJac;                                                    \
    }                                                                        \
  }

DirectionFromParallaxAngleSubTest(dN_dp_WCmi, Eigen::Matrix3d, 1e-6)

DirectionFromParallaxAngleSubTest(dN_dtheta_WCmi, Eigen::Matrix3d, 5e-6)

DirectionFromParallaxAngleSubTest(dN_dp_WCai, Eigen::Matrix3d, 1e-6)

DirectionFromParallaxAngleSubTest(dN_dp_WCtij, Eigen::Matrix3d, 1e-6)

DirectionFromParallaxAngleSubTest(dN_dthetai, Eigen::Vector3d, 1e-6)

typedef Eigen::Matrix<double,3,2> M32D;
DirectionFromParallaxAngleSubTest(dN_dni, M32D, 5e-6)

TEST(DirectionFromParallaxAngleAssociate, dN_dp_WCai) {
  simul::SimulatedMotionForParallaxAngleTest simulatedMotion(
      simul::MotionType::AssociateObserver, false);
  Eigen::Matrix3d numericJac;
  simulatedMotion.dN_dp_WCai(&numericJac);

  Eigen::Matrix3d analyticJac;
  simulatedMotion.dfpaj_.dN_dp_WCai(&analyticJac);

  Eigen::Matrix3d numericJac2;
  simulatedMotion.dN_dp_WCtij(&numericJac2);

  Eigen::Matrix3d analyticJac2;
  simulatedMotion.dfpaj_.dN_dp_WCtij(&analyticJac2);

  EXPECT_LT((numericJac + numericJac2 - analyticJac - analyticJac2)
                .lpNorm<Eigen::Infinity>(),
            1e-6);
}
