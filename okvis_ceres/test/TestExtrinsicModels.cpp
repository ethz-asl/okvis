#include <gtest/gtest.h>

#include <msckf/ExtrinsicModels.hpp>

TEST(Extrinsic_p_CB, ominus) {
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  Eigen::Vector3d delta = Eigen::Vector3d::Random();
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_BC(T_BC.r(), T_BC.q());
  okvis::Extrinsic_p_CB::oplus(delta.data(), &pair_T_BC);

  okvis::kinematics::Transformation new_T_BC(pair_T_BC.first, pair_T_BC.second);
  Eigen::Vector3d new_delta;
  okvis::Extrinsic_p_CB::ominus(T_BC.parameters().data(),
                                new_T_BC.parameters().data(), new_delta.data());
  EXPECT_LT((delta - new_delta).lpNorm<Eigen::Infinity>(), 1e-6);
}

TEST(Extrinsic_p_BC_q_BC, ominus) {
  okvis::kinematics::Transformation T_BC;
  T_BC.setRandom();
  Eigen::Matrix<double, 6, 1> delta = Eigen::Matrix<double, 6, 1>::Random();
  std::pair<Eigen::Vector3d, Eigen::Quaterniond> pair_T_BC(T_BC.r(), T_BC.q());
  okvis::Extrinsic_p_BC_q_BC::oplus(delta.data(), &pair_T_BC);
  okvis::kinematics::Transformation new_T_BC(pair_T_BC.first, pair_T_BC.second);
  Eigen::Matrix<double, 6, 1> new_delta;

  okvis::Extrinsic_p_BC_q_BC::ominus(T_BC.parameters().data(),
                                     new_T_BC.parameters().data(), new_delta.data());

  EXPECT_LT((delta - new_delta).lpNorm<Eigen::Infinity>(), 1e-6);
}
