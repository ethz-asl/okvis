
/**
 * @file implementation/Estimator.hpp
 * @brief Header implementation file for the Estimator class.
 * @author Jianzhu Huai
 */

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
  if (camera_rig_.getProjectionOptMode(camIdx) == ProjectionOptFixed::kModelId) {
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
  } else {
    std::shared_ptr<ceres::ReprojectionError<GEOMETRY_TYPE>> reprojectionError(
        new ceres::ReprojectionError<GEOMETRY_TYPE>(cameraGeometry, camIdx,
                                                    measurement, information));

    ::ceres::ResidualBlockId retVal = mapPtr_->addResidualBlock(
        reprojectionError,
        cauchyLossFunctionPtr_ ? cauchyLossFunctionPtr_.get() : NULL,
        mapPtr_->parameterBlockPtr(poseId),
        mapPtr_->parameterBlockPtr(landmarkId),
        mapPtr_->parameterBlockPtr(statesMap_.at(poseId)
                                       .sensors.at(SensorStates::Camera)
                                       .at(camIdx)
                                       .at(CameraSensorStates::T_SCi)
                                       .id));
    return retVal;
    LOG(ERROR) << "Not implemented point residual factor!";
    return 0;
  }
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
}  // namespace okvis
