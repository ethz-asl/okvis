#ifndef INCLUDE_MSCKF_CHECK_SENSOR_RIG_HPP_
#define INCLUDE_MSCKF_CHECK_SENSOR_RIG_HPP_

#include <msckf/imu/ImuRig.hpp>
#include <msckf/CameraRig.hpp>

namespace okvis {
bool doesExtrinsicModelFitImuModel(std::string extrinsicModel,
                                   std::string imuModel);
}  // namespace okvis

#endif // INCLUDE_MSCKF_CHECK_SENSOR_RIG_HPP_
