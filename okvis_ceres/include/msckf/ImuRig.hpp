#ifndef INCLUDE_MSCKF_IMU_RIG_HPP_
#define INCLUDE_MSCKF_IMU_RIG_HPP_

#include <okvis/kinematics/Transformation.hpp>

namespace okvis {

class ImuRig {
  std::vector<std::vector<bool>> fixed_intrinsic_mask_;
  std::vector<int> model_type_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline ImuRig() {}

  inline int getModelType(int camera_id) {
    if ((int)model_type_.size() > camera_id) {
      return model_type_[camera_id];
    } else {
      return -1;
    }
  }

  inline int addImu(const std::vector<bool>& fixedIntrinsicMask = std::vector<bool>()) {
    fixed_intrinsic_mask_.emplace_back(fixedIntrinsicMask);
    model_type_.push_back(0);
    return static_cast<int>(model_type_.size()) - 1;
  }
};

}  // namespace okvis
#endif  // INCLUDE_MSCKF_IMU_RIG_HPP_
