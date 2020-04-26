#include <msckf/ExtrinsicModels.hpp>
#include <msckf/TransformMultiplyJacobian.hpp>

namespace okvis {

void Extrinsic_p_C0C_q_C0C::dT_BC_dExtrinsic(
    const okvis::kinematics::Transformation& T_BCi,
    const okvis::kinematics::Transformation& T_BC0,
    Eigen::Matrix<double, 6, kNumParams>* j_C0Ci,
    Eigen::Matrix<double, 6, kNumParams>* j_BC0) {
  msckf::TransformMultiplyJacobian tmj(T_BC0, T_BC0.inverse() * T_BCi);
  j_BC0->topLeftCorner<3, 3>() = tmj.dp_dp_AB();
  j_BC0->topRightCorner<3, 3>() = tmj.dp_dtheta_AB();
  j_BC0->bottomLeftCorner<3, 3>() = tmj.dtheta_dp_AB();
  j_BC0->bottomRightCorner<3, 3>() = tmj.dtheta_dtheta_AB();

  j_C0Ci->topLeftCorner<3, 3>() = tmj.dp_dp_BC();
  j_C0Ci->topRightCorner<3, 3>() = tmj.dp_dtheta_BC();
  j_C0Ci->bottomLeftCorner<3, 3>() = tmj.dtheta_dp_BC();
  j_C0Ci->bottomRightCorner<3, 3>() = tmj.dtheta_dtheta_BC();
}
}  // namespace okvis
