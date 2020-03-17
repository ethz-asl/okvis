#ifndef INCLUDE_MSCKF_CHECK_SENSOR_RIG_HPP_
#define INCLUDE_MSCKF_CHECK_SENSOR_RIG_HPP_

#include <msckf/imu/ImuRig.hpp>
#include <msckf/CameraRig.hpp>

namespace okvis {
  // check whether the parameter settings are observable
  bool checkSensorRigObservability(
      const cameras::CameraRig& cam_rig, const ImuRig& imu_rig);

  // used in the point-frame residual to construct the camera geometry
  // from a list of parameters
  std::shared_ptr<cameras::CameraBase> constructCameraGeometry(
      const int camera_model_id, const double** parameters);

  // used in Imu residual to construct the imu model for
  // correcting the imu intrinsic errors
  // from a list of parameters
//  std::shared_ptr<ImuModelBase> constructImuModel(
//      const int imu_model_id, const double** parameters);
}

#endif // INCLUDE_MSCKF_CHECK_SENSOR_RIG_HPP_
