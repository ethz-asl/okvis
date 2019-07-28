
/**
 * @file implementation/Estimator.hpp
 * @brief Header implementation file for the Estimator class.
 * @author Jianzhu Huai
 */

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
  if (camera_rig_.getModelType(camIdx) == -1) {
    // create error term
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
    // create a reprojection error using the model id from the camera_rig_,
    // and measurement, and uncertainty

    // add residual block with relevant parameter blocks,
    // these parameter blocks can be constant or variable and will be
    // used to construct the camera geometry model in the reprojection residual
    return 0;
  }
}
}  // namespace okvis
