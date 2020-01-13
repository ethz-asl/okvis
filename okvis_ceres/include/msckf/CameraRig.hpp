#ifndef INCLUDE_MSCKF_CAMERA_RIG_HPP_
#define INCLUDE_MSCKF_CAMERA_RIG_HPP_

#include <map>

#include <msckf/ExtrinsicModels.hpp>
#include <msckf/ProjParamOptModels.hpp>
#include <okvis/cameras/CameraBase.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/FovDistortion.hpp>
#include <okvis/cameras/NoDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion.hpp>
#include <okvis/cameras/RadialTangentialDistortion8.hpp>

#include <okvis/kinematics/Transformation.hpp>
#include <glog/logging.h>

namespace okvis {
namespace cameras {

std::shared_ptr<cameras::CameraBase> cloneCameraGeometry(
    std::shared_ptr<const cameras::CameraBase> cameraGeometry);

static inline void DistortionNameToParamsInfo(const std::string& distortionName,
                                              const std::string delimiter,
                                              std::string* paramsInfo) {
  std::map<std::string, std::string> distortionNameList{
      {okvis::cameras::EquidistantDistortion().type(),
       "k1" + delimiter + "k2" + delimiter + "k3" + delimiter + "k4"},
      {okvis::cameras::RadialTangentialDistortion().type(),
       "k1" + delimiter + "k2" + delimiter + "p1" + delimiter + "p2"},
      {okvis::cameras::NoDistortion().type(), ""},
      {okvis::cameras::RadialTangentialDistortion8().type(),
       "k1" + delimiter + "k2" + delimiter + "p1" + delimiter + "p2" +
           delimiter + "k3" + delimiter + "k4" + delimiter + "k5" + delimiter +
           "k6"},
      {okvis::cameras::FovDistortion().type(), "omega"}};

  std::map<std::string, std::string>::iterator it = std::find_if(
      distortionNameList.begin(), distortionNameList.end(),
      [&distortionName](const std::pair<std::string, std::string>& val) {
        if (val.first.compare(distortionName) == 0) return true;
        return false;
      });
  if (it == distortionNameList.end()) {
    *paramsInfo = "";
  } else {
    *paramsInfo = it->second;
  }
}

class CameraRig {
 private:
  ///< Mounting transformations from IMU
  std::vector<std::shared_ptr<okvis::kinematics::Transformation>> T_SC_;
  ///< Camera geometries
  std::vector<std::shared_ptr<cameras::CameraBase>> camera_geometries_;

  ///< which subset of the extrinsic parameters are variable in estimation.
  std::vector<int> extrinsic_opt_rep_;

  ///< which subset of the projection intrinsic parameters are variable in estimation.
  std::vector<int> proj_opt_rep_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline CameraRig() {}
  inline void clear() {
    T_SC_.clear();
    camera_geometries_.clear();
    extrinsic_opt_rep_.clear();
    proj_opt_rep_.clear();
  }
  inline double getImageDelay(int camera_id) const {
    return camera_geometries_[camera_id]->imageDelay();
  }
  inline double getReadoutTime(int camera_id) const {
    return camera_geometries_[camera_id]->readoutTime();
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
  inline std::shared_ptr<const cameras::CameraBase> getCameraGeometry(
      int camera_id) const {
    return camera_geometries_[camera_id];
  }
  inline std::shared_ptr<cameras::CameraBase> getMutableCameraGeometry(
      int camera_id) const {
    return camera_geometries_[camera_id];
  }
  inline int getIntrinsicDimen(int camera_id) const {
    return camera_geometries_[camera_id]->noIntrinsicsParameters();
  }
  inline int getProjectionOptMode(int camera_id) const {
    if (camera_id >= static_cast<int>(proj_opt_rep_.size())) {
      return ProjectionOptFixed::kModelId;
    } else {
      return proj_opt_rep_[camera_id];
    }
  }
  inline int getExtrinsicOptMode(int camera_id) const {
    return extrinsic_opt_rep_[camera_id];
  }
  inline int getDistortionDimen(int camera_id) const {
    return camera_geometries_[camera_id]->noDistortionParameters();
  }
  inline int getMinimalExtrinsicDimen(int camera_id) const {
    return ExtrinsicModelGetMinimalDim(extrinsic_opt_rep_[camera_id]);
  }
  inline int getMinimalProjectionDimen(int camera_id) const {
    return ProjectionOptGetMinimalDim(proj_opt_rep_[camera_id]);
  }
  inline int getCameraParamsMinimalDim(int camera_id) const {
    std::shared_ptr<okvis::cameras::CameraBase> camera =
        camera_geometries_[camera_id];
    return getMinimalExtrinsicDimen(camera_id) +
           getMinimalProjectionDimen(camera_id) +
           camera->noDistortionParameters() + 2;  // 2 for td and tr
  }
  inline void setImageDelay(int camera_id, double td) {
    camera_geometries_[camera_id]->setImageDelay(td);
  }
  inline void setReadoutTime(int camera_id, double tr) {
    camera_geometries_[camera_id]->setReadoutTime(tr);
  }
  inline void setCameraExtrinsic(
      int camera_id, const okvis::kinematics::Transformation& T_SC) {
    *(T_SC_[camera_id]) = T_SC;
  }
  inline void setProjectionOptMode(int opt_mode, int camera_id) {
    proj_opt_rep_[camera_id] = opt_mode;
  }
  inline void setExtrinsicOptMode(int opt_mode, int camera_id) {
    extrinsic_opt_rep_[camera_id] = opt_mode;
  }
  inline void setCameraIntrinsics(int camera_id,
                                  const Eigen::VectorXd& intrinsic_vec) {
    camera_geometries_[camera_id]->setIntrinsics(intrinsic_vec);
  }
  inline void setCameraIntrinsics(int camera_id,
                                  const Eigen::VectorXd& projection_vec,
                                  const Eigen::VectorXd& distortion_vec) {
    Eigen::VectorXd intrinsicParameters;
    camera_geometries_[camera_id]->getIntrinsics(intrinsicParameters);
    const int distortionDim =
        camera_geometries_[camera_id]->noDistortionParameters();
    intrinsicParameters.tail(distortionDim) = distortion_vec;
    ProjectionOptLocalToGlobal(proj_opt_rep_[camera_id], projection_vec,
                               &intrinsicParameters);

    camera_geometries_[camera_id]->setIntrinsics(intrinsicParameters);
  }
  inline int addCamera(
      std::shared_ptr<const okvis::kinematics::Transformation> T_SC,
      std::shared_ptr<const cameras::CameraBase> cameraGeometry,
      std::string pinhole_opt_rep, std::string extrinsic_opt_rep) {
    T_SC_.emplace_back(
        std::make_shared<okvis::kinematics::Transformation>(*T_SC));
    camera_geometries_.emplace_back(cloneCameraGeometry(cameraGeometry));
    proj_opt_rep_.emplace_back(ProjectionOptNameToId(pinhole_opt_rep));
    LOG(INFO) << "added proj opt rep " << pinhole_opt_rep << " of id "
              << proj_opt_rep_.back();
    extrinsic_opt_rep_.emplace_back(ExtrinsicModelNameToId(extrinsic_opt_rep));
    LOG(INFO) << "added extrinsic opt rep " << extrinsic_opt_rep << " of id "
              << extrinsic_opt_rep_.back();
    return static_cast<int>(T_SC_.size()) - 1;
  }
};
}  // namespace cameras
}  // namespace okvis
#endif  // INCLUDE_MSCKF_CAMERA_RIG_HPP_
