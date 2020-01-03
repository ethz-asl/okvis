#include <msckf/CameraRig.hpp>

#include <glog/logging.h>

#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>
#include <okvis/cameras/FovDistortion.hpp>

namespace okvis {
namespace cameras {

std::shared_ptr<cameras::CameraBase> cloneCameraGeometry(
    std::shared_ptr<const cameras::CameraBase> cameraGeometry) {
  std::string geometryType = cameraGeometry->type();
  std::string distortionType = cameraGeometry->distortionType();
  Eigen::VectorXd intrinsic_vec;
  cameraGeometry->getIntrinsics(intrinsic_vec);
  uint64_t id = cameraGeometry->id();
  if (geometryType.find("PinholeCamera<") == 0) {
    const int distortion_start_index = 4;
    if (strcmp(distortionType.c_str(), "EquidistantDistortion") == 0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::EquidistantDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::EquidistantDistortion(
                  intrinsic_vec[distortion_start_index],
                  intrinsic_vec[distortion_start_index + 1],
                  intrinsic_vec[distortion_start_index + 2],
                  intrinsic_vec[distortion_start_index + 3]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));

    } else if (strcmp(distortionType.c_str(), "RadialTangentialDistortion") ==
               0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::RadialTangentialDistortion(
                  intrinsic_vec[distortion_start_index],
                  intrinsic_vec[distortion_start_index + 1],
                  intrinsic_vec[distortion_start_index + 2],
                  intrinsic_vec[distortion_start_index + 3]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));

    } else if (strcmp(distortionType.c_str(), "RadialTangentialDistortion8") ==
               0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::RadialTangentialDistortion8>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::RadialTangentialDistortion8(
                  intrinsic_vec[distortion_start_index],
                  intrinsic_vec[distortion_start_index + 1],
                  intrinsic_vec[distortion_start_index + 2],
                  intrinsic_vec[distortion_start_index + 3],
                  intrinsic_vec[distortion_start_index + 4],
                  intrinsic_vec[distortion_start_index + 5],
                  intrinsic_vec[distortion_start_index + 6],
                  intrinsic_vec[distortion_start_index + 7]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));
    } else if (strcmp(distortionType.c_str(), "NoDistortion") == 0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<okvis::cameras::NoDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3], okvis::cameras::NoDistortion(),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));
    } else if (strcmp(distortionType.c_str(), "FovDistortion") == 0) {
      return std::shared_ptr<okvis::cameras::CameraBase>(
          new okvis::cameras::PinholeCamera<
              okvis::cameras::FovDistortion>(
              cameraGeometry->imageWidth(), cameraGeometry->imageHeight(),
              intrinsic_vec[0], intrinsic_vec[1], intrinsic_vec[2],
              intrinsic_vec[3],
              okvis::cameras::FovDistortion(
                  intrinsic_vec[distortion_start_index]),
              cameraGeometry->imageDelay(), cameraGeometry->readoutTime(),
              id));
    } else {
      LOG(ERROR) << "unrecognized distortion type " << distortionType;
    }
  } else {
    LOG(ERROR) << "unrecognized camera geometry type "
               << cameraGeometry->type();
  }
  return std::shared_ptr<cameras::CameraBase>();
}

}  // namespace cameras
}  // namespace okvis
