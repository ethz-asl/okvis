
/**
 * @file implementation/Estimator.hpp
 * @brief Header implementation file for the Estimator class.
 * @author Jianzhu Huai
 */

#include <msckf/EpipolarFactor.hpp>
#include <msckf/ProjParamOptModels.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

template<class GEOMETRY_TYPE>
::ceres::ResidualBlockId Estimator::addPointFrameResidual(
    uint64_t landmarkId,
    uint64_t poseId,
    size_t camIdx,
    const Eigen::Vector2d& measurement,
    const Eigen::Matrix2d& information,
    std::shared_ptr<const GEOMETRY_TYPE> cameraGeometry) {
  // TODO(jhuai): add residual according to
  // (camera_rig_.getProjectionOptMode(camIdx) == ProjectionOptFixed::kModelId)
  std::shared_ptr < ceres::ReprojectionError
      < GEOMETRY_TYPE
          >> reprojectionError(
              new ceres::ReprojectionError<GEOMETRY_TYPE>(
                  cameraGeometry,
                  camIdx, measurement, information));

  ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
      reprojectionError,
      cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,
      mapPtr_->parameterBlockPtr(poseId),
      mapPtr_->parameterBlockPtr(landmarkId),
      mapPtr_->parameterBlockPtr(
          statesMap_.at(poseId).sensors.at(SensorStates::Camera).at(camIdx).at(
              CameraSensorStates::T_SCi).id));
  return retVal;
}

template<class PARAMETER_BLOCK_T>
bool Estimator::getSensorStateEstimateAs(
    uint64_t poseId, int sensorIdx, int sensorType, int stateType,
    typename PARAMETER_BLOCK_T::estimate_t & state) const
{
#if 0
  PARAMETER_BLOCK_T stateParameterBlock;
  if (!getSensorStateParameterBlockAs(poseId, sensorIdx, sensorType, stateType,
                                      stateParameterBlock)) {
    return false;
  }
  state = stateParameterBlock.estimate();
  return true;
#else
  // convert base class pointer with various levels of checking
  std::shared_ptr<ceres::ParameterBlock> parameterBlockPtr;
  if (!getSensorStateParameterBlockPtr(poseId, sensorIdx, sensorType, stateType,
                                       parameterBlockPtr)) {
      return false;
  }
#ifndef NDEBUG
  std::shared_ptr<PARAMETER_BLOCK_T> derivedParameterBlockPtr =
          std::dynamic_pointer_cast<PARAMETER_BLOCK_T>(parameterBlockPtr);
  if(!derivedParameterBlockPtr) {
      std::shared_ptr<PARAMETER_BLOCK_T> info(new PARAMETER_BLOCK_T);
      OKVIS_THROW_DBG(Exception,"wrong pointer type requested: requested "
                      <<info->typeInfo()<<" but is of type"
                      <<parameterBlockPtr->typeInfo())
              return false;
  }
  state = derivedParameterBlockPtr->estimate();
#else
  state = std::static_pointer_cast<PARAMETER_BLOCK_T>(
              parameterBlockPtr)->estimate();
#endif
  return true;
#endif
}

template <class CAMERA_GEOMETRY_T>
bool Estimator::replaceEpipolarWithReprojectionErrors(uint64_t lmId) {
  PointMap::iterator lmIt = landmarksMap_.find(lmId);
  std::map<okvis::KeypointIdentifier, uint64_t>& obsMap =
      lmIt->second.observations;
  for (std::map<okvis::KeypointIdentifier, uint64_t>::iterator obsIter = obsMap.begin();
       obsIter != obsMap.end(); ++obsIter) {
    if (obsIter->second != 0u) {
      mapPtr_->removeResidualBlock(
          reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
    }
    const KeypointIdentifier& kpi = obsIter->first;
    // get the keypoint measurement
    okvis::MultiFramePtr multiFramePtr = multiFramePtrMap_.at(kpi.frameId);
    Eigen::Vector2d measurement;
    multiFramePtr->getKeypoint(kpi.cameraIndex, kpi.keypointIndex, measurement);
    Eigen::Matrix2d information = Eigen::Matrix2d::Identity();
    double size = 1.0;
    multiFramePtr->getKeypointSize(kpi.cameraIndex, kpi.keypointIndex, size);
    information *= 64.0 / (size * size);

    std::shared_ptr<const okvis::cameras::CameraBase> baseCameraGeometry =
        camera_rig_.getCameraGeometry(kpi.cameraIndex);
    std::shared_ptr<const CAMERA_GEOMETRY_T> argCameraGeometry =
        std::static_pointer_cast<const CAMERA_GEOMETRY_T>(baseCameraGeometry);
    ::ceres::ResidualBlockId retVal = addPointFrameResidual(
        lmId, kpi.frameId, kpi.cameraIndex,
        measurement, information, argCameraGeometry);
    obsIter->second = reinterpret_cast<uint64_t>(retVal);
  }
  return true;
}

template <class CAMERA_GEOMETRY_T>
bool Estimator::addEpipolarConstraint(uint64_t landmarkId, uint64_t poseId,
                                      size_t camIdx, size_t keypointIdx,
                                      bool removeExisting) {
  PointMap::iterator lmkIt = landmarksMap_.find(landmarkId);
  if (lmkIt == landmarksMap_.end())
    return false;
  okvis::KeypointIdentifier kidTail(poseId, camIdx, keypointIdx);

  // avoid double observations
  if (lmkIt->second.observations.find(kidTail) !=
      lmkIt->second.observations.end()) {
    return false;
  }

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      measurement12(2);
  std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d>>
      covariance12(2);
  // get the head keypoint measurement
  const okvis::KeypointIdentifier& kidHead =
      lmkIt->second.observations.begin()->first;
  okvis::MultiFramePtr multiFramePtr =
      multiFramePtrMap_.at(kidHead.frameId);
  multiFramePtr->getKeypoint(kidHead.cameraIndex, kidHead.keypointIndex,
                             measurement12[0]);
  covariance12[0] = Eigen::Matrix2d::Identity();
  double size = 1.0;
  multiFramePtr->getKeypointSize(kidHead.cameraIndex,
                                 kidHead.keypointIndex, size);
  covariance12[0] *= (size * size) / 64.0;

  // get the tail keypoint measurement
  multiFramePtr = multiFramePtrMap_.at(poseId);
  multiFramePtr->getKeypoint(camIdx, keypointIdx, measurement12[1]);
  covariance12[1] = Eigen::Matrix2d::Identity();
  size = 1.0;
  multiFramePtr->getKeypointSize(camIdx, keypointIdx, size);
  covariance12[1] *= (size * size) / 64.0;

  std::shared_ptr<const okvis::cameras::CameraBase> baseCameraGeometry =
      camera_rig_.getCameraGeometry(camIdx);
  std::shared_ptr<const CAMERA_GEOMETRY_T> argCameraGeometry =
      std::static_pointer_cast<const CAMERA_GEOMETRY_T>(baseCameraGeometry);

  auto& stateLeft = statesMap_.at(kidHead.frameId);
  auto& stateRight = statesMap_.at(poseId);

  std::vector<okvis::Time> stateEpoch = {stateLeft.timestamp,
                                         stateRight.timestamp};
  std::vector<okvis::ImuMeasurementDeque,
              Eigen::aligned_allocator<okvis::ImuMeasurementDeque>>
      imuMeasCanopy = {
          inertialMeasForStates_.findWindow(stateEpoch[0], half_window_),
          inertialMeasForStates_.findWindow(stateEpoch[1], half_window_)};
  okvis::kinematics::Transformation T_SC_base =
      camera_rig_.getCameraExtrinsic(camIdx);

  std::vector<double> tdAtCreation = {stateLeft.tdAtCreation.toSec(),
                                      stateRight.tdAtCreation.toSec()};

  std::vector<okvis::SpeedAndBias,
              Eigen::aligned_allocator<okvis::SpeedAndBias>>
      speedAndBias12(2);
  // The below speed and bias block may not exist because marginalization step
  // may marginalize speed and biases without removing poses.
  // TODO(jhuai): fix this issue by always marg speed and bias along with
  // keyframes or discard them along with frames as in VINS-Mono
//  uint64_t sbId = stateLeft.sensors.at(SensorStates::Imu)
//                      .at(0)
//                      .at(ImuSensorStates::SpeedAndBias)
//                      .id;
//  speedAndBias12[0] =
//      std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
//          mapPtr_->parameterBlockPtr(sbId))
//          ->estimate();
  uint64_t sbId = stateRight.sensors.at(SensorStates::Imu)
             .at(0)
             .at(ImuSensorStates::SpeedAndBias)
             .id;
  speedAndBias12[1] =
      std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
          mapPtr_->parameterBlockPtr(sbId))
          ->estimate();
  speedAndBias12[0] = speedAndBias12[1];

  double gravityMag = imuParametersVec_.at(0).g;

  std::shared_ptr<::ceres::CostFunction> twoViewError;
  switch (camera_rig_.getProjectionOptMode(camIdx)) {
  case ProjectionOptFixed::kModelId:
  case ProjectionOptFXY_CXY::kModelId:
      twoViewError.reset(new ceres::EpipolarFactor<CAMERA_GEOMETRY_T, Extrinsic_p_BC_q_BC,
                         ProjectionOptFXY_CXY>(
                             argCameraGeometry, landmarkId, measurement12, covariance12,
                             imuMeasCanopy, T_SC_base, stateEpoch, tdAtCreation,
                             speedAndBias12, gravityMag));
      break;
  case ProjectionOptFX_CXY::kModelId:
      twoViewError.reset(new ceres::EpipolarFactor<CAMERA_GEOMETRY_T, Extrinsic_p_BC_q_BC,
                         ProjectionOptFX_CXY>(
                             argCameraGeometry, landmarkId, measurement12, covariance12,
                             imuMeasCanopy, T_SC_base, stateEpoch, tdAtCreation,
                             speedAndBias12, gravityMag));
      break;
  case ProjectionOptFX::kModelId:
      twoViewError.reset(new ceres::EpipolarFactor<CAMERA_GEOMETRY_T, Extrinsic_p_BC_q_BC,
                         ProjectionOptFX>(
                             argCameraGeometry, landmarkId, measurement12, covariance12,
                             imuMeasCanopy, T_SC_base, stateEpoch, tdAtCreation,
                             speedAndBias12, gravityMag));
      break;
  default:
      break;
  }

  ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
      twoViewError,
      cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,
      mapPtr_->parameterBlockPtr(kidHead.frameId),
      mapPtr_->parameterBlockPtr(poseId),
      mapPtr_->parameterBlockPtr(stateRight.sensors.at(SensorStates::Camera)
                                     .at(camIdx)
                                     .at(CameraSensorStates::T_SCi)
                                     .id),
      mapPtr_->parameterBlockPtr(stateRight.sensors.at(SensorStates::Camera)
                                     .at(camIdx)
                                     .at(CameraSensorStates::Intrinsics)
                                     .id),
      mapPtr_->parameterBlockPtr(stateRight.sensors.at(SensorStates::Camera)
                                     .at(camIdx)
                                     .at(CameraSensorStates::Distortion)
                                     .id),
      mapPtr_->parameterBlockPtr(stateRight.sensors.at(SensorStates::Camera)
                                     .at(camIdx)
                                     .at(CameraSensorStates::TR)
                                     .id),
      mapPtr_->parameterBlockPtr(stateRight.sensors.at(SensorStates::Camera)
                                     .at(camIdx)
                                     .at(CameraSensorStates::TD)
                                     .id));

  if (removeExisting) {
    for (auto obsIter = lmkIt->second.observations.begin();
         obsIter != lmkIt->second.observations.end(); ++obsIter) {
      if (obsIter->second != 0) {
        mapPtr_->removeResidualBlock(
            reinterpret_cast<::ceres::ResidualBlockId>(obsIter->second));
        obsIter->second = 0;
      }
    }
  }

  // remember
  lmkIt->second.observations.insert(
      std::pair<okvis::KeypointIdentifier, uint64_t>(
          kidTail, reinterpret_cast<uint64_t>(retVal)));

  return true;
}
}  // namespace okvis
