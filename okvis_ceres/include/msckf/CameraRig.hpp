#ifndef INCLUDE_MSCKF_CAMERA_RIG_HPP_
#define INCLUDE_MSCKF_CAMERA_RIG_HPP_
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
namespace cameras {

std::shared_ptr<cameras::CameraBase> cloneCameraGeometry(
    std::shared_ptr<const cameras::CameraBase> cameraGeometry);

class CameraRig {
 private:
  ///< Mounting transformations from IMU
  std::vector<std::shared_ptr<okvis::kinematics::Transformation>> T_SC_;
  ///< Camera geometries
  std::vector<std::shared_ptr<cameras::CameraBase>> camera_geometries_;
  // for each intrinsic parameter of a camera, if 1 fixed, if 0 float to be
  // optimized. If empty, all to be optimized
  std::vector<std::vector<bool>> fixed_intrinsic_mask_;
  ///< time in secs to read out a frame, applies to rolling shutter cameras
  std::vector<double> frame_readout_time_;
  ///< at the same epoch, timestamp by camera_clock + time_delay =
  ///  timestamp by IMU clock
  std::vector<double> time_delay_;

  std::vector<int> model_type_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline CameraRig() {}
  inline double getTimeDelay(int camera_id) const {
    return time_delay_[camera_id];
  }
  inline double getReadoutTime(int camera_id) const {
    return frame_readout_time_[camera_id];
  }
  inline uint32_t getImageWidth(int camera_id) const {
    return camera_geometries_[camera_id]->imageWidth();
  }
  inline uint32_t getImageHeight(int camera_id) const {
    return camera_geometries_[camera_id]->imageHeight();
  }
  inline okvis::kinematics::Transformation getCameraExtrinsic(
      int camera_id) const {
    return *(T_SC_[camera_id]);
  }
  inline std::shared_ptr<cameras::CameraBase> getCameraGeometry(
      int camera_id) const {
    return camera_geometries_[camera_id];
  }

  inline int getModelType(int camera_id) {
    if ((int)model_type_.size() > camera_id) {
      return model_type_[camera_id];
    } else {
      return -1;
    }
  }

  inline void setTimeDelay(int camera_id, double td) {
    time_delay_[camera_id] = td;
  }
  inline void setReadoutTime(int camera_id, double tr) {
    frame_readout_time_[camera_id] = tr;
  }
  inline void setCameraExtrinsic(
      int camera_id, const okvis::kinematics::Transformation& T_SC) {
    *(T_SC_[camera_id]) = T_SC;
  }

  inline void setCameraIntrinsics(int camera_id,
                                  const Eigen::VectorXd& intrinsic_vec) {
    camera_geometries_[camera_id]->setIntrinsics(intrinsic_vec);
  }

  inline int addCamera(
      std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
      std::shared_ptr<const cameras::CameraBase> cameraGeometry, double tr,
      double td,
      const std::vector<bool>& fixedIntrinsicMask = std::vector<bool>()) {
    T_SC_.emplace_back(
        std::make_shared<okvis::kinematics::Transformation>(*T_SC));
    camera_geometries_.emplace_back(cloneCameraGeometry(cameraGeometry));
    frame_readout_time_.emplace_back(tr);
    time_delay_.emplace_back(td);
    fixed_intrinsic_mask_.emplace_back(fixedIntrinsicMask);
    return static_cast<int>(T_SC_.size()) - 1;
  }
};
}  // namespace cameras
}  // namespace okvis
#endif  // INCLUDE_MSCKF_CAMERA_RIG_HPP_
