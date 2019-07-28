#include <msckf/checkSensorRig.hpp>

namespace okvis {
  bool checkSensorRigObservability(
      const cameras::CameraRig& cam_rig, const ImuRig& imu_rig) {
    return true;
  }

  std::shared_ptr<cameras::CameraBase> constructCameraGeometry(
      const int camera_model_id, const double** parameters) {
    return nullptr;
  }
}
