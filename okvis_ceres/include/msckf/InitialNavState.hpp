#ifndef INITIAL_PV_AND_STD_HPP
#define INITIAL_PV_AND_STD_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>  //quaterniond
#include <okvis/Parameters.hpp>

namespace okvis {

struct InitialNavState {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // S represents the nominal IMU sensor frame realized with the camera frame
  // and the intersection of three accelerometers W represents the world frame
  // with z along the negative gravity direction and has minimal rotation
  // relative to the S frame at the initialization epoch
  bool initWithExternalSource_;
  okvis::Time stateTime;  // epoch for the initialization values
  Eigen::Vector3d p_WS;
  Eigen::Quaterniond q_WS;
  Eigen::Vector3d v_WS;
  Eigen::Vector3d std_p_WS;
  Eigen::Vector3d
      std_q_WS;  // std of $\delta \theta$ which is expressed in the world frame
  Eigen::Vector3d std_v_WS;

  InitialNavState();

  InitialNavState(const okvis::InitialState& rhs);

  void updatePose(const okvis::kinematics::Transformation& T_WS,
                  const okvis::Time state_time);

  InitialNavState& operator=(const InitialNavState& other);
};

}  // namespace okvis
#endif  // INITIAL_PV_AND_STD_HPP
