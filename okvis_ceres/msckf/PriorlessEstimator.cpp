/**
 * @file PriorlessEstimator.cpp
 * @brief Source file for the PriorlessEstimator class.
 * @author Jianzhu Huai
 */

#include <msckf/EuclideanParamBlock.hpp>
#include <msckf/PriorlessEstimator.hpp>
#include <msckf/CameraTimeParamBlock.hpp>

#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/assert_macros.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

// Constructor if a ceres map is already available.
PriorlessEstimator::PriorlessEstimator(
    std::shared_ptr<okvis::ceres::Map> mapPtr)
    : Estimator(mapPtr)
{
}

// The default constructor.
PriorlessEstimator::PriorlessEstimator()
    : Estimator()
{
}

PriorlessEstimator::~PriorlessEstimator()
{
}

// Add a pose to the state.
bool PriorlessEstimator::addStates(
    okvis::MultiFramePtr multiFrame,
    const okvis::ImuMeasurementDeque & imuMeasurements,
    bool asKeyframe)
{
  // note: this is before matching...
  // TODO !!
  okvis::kinematics::Transformation T_WS;
  okvis::SpeedAndBias speedAndBias;
  // record the imu measurements between two consecutive states
  inertialMeasForStates_.push_back(imuMeasurements);
  if (statesMap_.empty()) {
    // in case this is the first frame ever, let's initialize the pose:
    if (pvstd_.initWithExternalSource)
      T_WS = okvis::kinematics::Transformation(pvstd_.p_WS, pvstd_.q_WS);
    else {
      bool success0 = initPoseFromImu(imuMeasurements, T_WS);
      OKVIS_ASSERT_TRUE_DBG(
          Exception, success0,
          "pose could not be initialized from imu measurements.");
      if (!success0) return false;
      pvstd_.updatePose(T_WS, multiFrame->timestamp());
    }
    speedAndBias.setZero();
    speedAndBias.head<3>() = pvstd_.v_WS;
    speedAndBias.segment<3>(6) = imuParametersVec_.at(0).a0;
  } else {
    // get the previous states
    uint64_t T_WS_id = statesMap_.rbegin()->second.id;
    uint64_t speedAndBias_id = statesMap_.rbegin()->second.sensors.at(SensorStates::Imu)
        .at(0).at(ImuSensorStates::SpeedAndBias).id;
    OKVIS_ASSERT_TRUE_DBG(Exception, mapPtr_->parameterBlockExists(T_WS_id),
                       "this is an okvis bug. previous pose does not exist.");
    T_WS = std::static_pointer_cast<ceres::PoseParameterBlock>(
        mapPtr_->parameterBlockPtr(T_WS_id))->estimate();
    //OKVIS_ASSERT_TRUE_DBG(
    //    Exception, speedAndBias_id,
    //    "this is an okvis bug. previous speedAndBias does not exist.");
    speedAndBias =
        std::static_pointer_cast<ceres::SpeedAndBiasParameterBlock>(
            mapPtr_->parameterBlockPtr(speedAndBias_id))->estimate();

    // propagate pose and speedAndBias
    int numUsedImuMeasurements = ceres::ImuError::propagation(
        imuMeasurements, imuParametersVec_.at(0), T_WS, speedAndBias,
        statesMap_.rbegin()->second.timestamp, multiFrame->timestamp());
    OKVIS_ASSERT_TRUE_DBG(Exception, numUsedImuMeasurements > 1,
                       "propagation failed");
    if (numUsedImuMeasurements < 1){
      LOG(INFO) << "numUsedImuMeasurements=" << numUsedImuMeasurements;
      return false;
    }
    okvis::Time secondLatestStateTime = statesMap_.rbegin()->second.timestamp;
    auto imuMeasCoverSecond = inertialMeasForStates_.findWindow(secondLatestStateTime, half_window_);
    statesMap_.rbegin()->second.imuReadingWindow.reset(new okvis::ImuMeasurementDeque(imuMeasCoverSecond));
  }


  // create a states object:
  States states(asKeyframe, multiFrame->id(), multiFrame->timestamp());

  // check if id was used before
  OKVIS_ASSERT_TRUE_DBG(Exception,
      statesMap_.find(states.id)==statesMap_.end(),
      "pose ID" <<states.id<<" was used before!");

  // create global states
  std::shared_ptr<okvis::ceres::PoseParameterBlock> poseParameterBlock(
      new okvis::ceres::PoseParameterBlock(T_WS, states.id,
                                           multiFrame->timestamp()));
  states.global.at(GlobalStates::T_WS).exists = true;
  states.global.at(GlobalStates::T_WS).id = states.id;
  auto imuMeasCover = inertialMeasForStates_.findWindow(multiFrame->timestamp(), half_window_);
  states.imuReadingWindow.reset(new okvis::ImuMeasurementDeque(imuMeasCover));
  if(statesMap_.empty()) {
    referencePoseId_ = states.id; // set this as reference pose
  }
  mapPtr_->addParameterBlock(poseParameterBlock,ceres::Map::Pose6d);

  // add to buffer
  statesMap_.insert(std::pair<uint64_t, States>(states.id, states));
  multiFramePtrMap_.insert(std::pair<uint64_t, okvis::MultiFramePtr>(states.id, multiFrame));

  // the following will point to the last states:
  std::map<uint64_t, States>::reverse_iterator lastElementIterator = statesMap_.rbegin();
  lastElementIterator++;

  // initialize new sensor states
  // cameras:
  for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {

    SpecificSensorStatesContainer cameraInfos(2);
    cameraInfos.at(CameraSensorStates::T_SCi).exists=true;
    cameraInfos.at(CameraSensorStates::Intrinsics).exists=false;
    if(((extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_translation<1e-12)||
        (extrinsicsEstimationParametersVec_.at(i).sigma_c_relative_orientation<1e-12))&&
        (statesMap_.size() > 1)){
      // use the same block...
      cameraInfos.at(CameraSensorStates::T_SCi).id =
          lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id;
    } else {
      const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
      uint64_t id = IdProvider::instance().newId();
      std::shared_ptr<okvis::ceres::PoseParameterBlock> extrinsicsParameterBlockPtr(
          new okvis::ceres::PoseParameterBlock(T_SC, id,
                                               multiFrame->timestamp()));
      if(!mapPtr_->addParameterBlock(extrinsicsParameterBlockPtr,ceres::Map::Pose6d)){
        return false;
      }
      cameraInfos.at(CameraSensorStates::T_SCi).id = id;
    }
    // update the states info
    statesMap_.rbegin()->second.sensors.at(SensorStates::Camera).push_back(cameraInfos);
    states.sensors.at(SensorStates::Camera).push_back(cameraInfos);
  }

  // IMU states are automatically propagated.
  for (size_t i=0; i<imuParametersVec_.size(); ++i){
    SpecificSensorStatesContainer imuInfo(2);
    imuInfo.at(ImuSensorStates::SpeedAndBias).exists = true;
    uint64_t id = IdProvider::instance().newId();
    std::shared_ptr<okvis::ceres::SpeedAndBiasParameterBlock> speedAndBiasParameterBlock(
        new okvis::ceres::SpeedAndBiasParameterBlock(speedAndBias, id, multiFrame->timestamp()));

    if(!mapPtr_->addParameterBlock(speedAndBiasParameterBlock)){
      return false;
    }
    imuInfo.at(ImuSensorStates::SpeedAndBias).id = id;
    statesMap_.rbegin()->second.sensors.at(SensorStates::Imu).push_back(imuInfo);
    states.sensors.at(SensorStates::Imu).push_back(imuInfo);
  }

  // depending on whether or not this is the very beginning, we will add priors or relative terms to the last state:
  if (statesMap_.size() == 1) {
    // let's add a prior
//    Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
//    information(5,5) = 1.0e8; information(0,0) = 1.0e8; information(1,1) = 1.0e8; information(2,2) = 1.0e8;
//    std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS, information));
//    /*auto id2= */ mapPtr_->addResidualBlock(poseError,NULL,poseParameterBlock);
    //mapPtr_->isJacobianCorrect(id2,1.0e-6);

    // sensor states
//    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
//      double translationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_translation;
//      double translationVariance = translationStdev*translationStdev;
//      double rotationStdev = extrinsicsEstimationParametersVec_.at(i).sigma_absolute_orientation;
//      double rotationVariance = rotationStdev*rotationStdev;
//      if(translationVariance>1.0e-16 && rotationVariance>1.0e-16){
//        const okvis::kinematics::Transformation T_SC = *multiFrame->T_SC(i);
//        std::shared_ptr<ceres::PoseError > cameraPoseError(
//              new ceres::PoseError(T_SC, translationVariance, rotationVariance));
//        // add to map
//        mapPtr_->addResidualBlock(
//            cameraPoseError,
//            NULL,
//            mapPtr_->parameterBlockPtr(
//                states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id));
//        //mapPtr_->isJacobianCorrect(id,1.0e-6);
//      }
//      else {
//        mapPtr_->setParameterBlockConstant(
//            states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id);
//      }
//    }
//    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
//      Eigen::Matrix<double,6,1> variances;
//      // get these from parameter file
//      const double sigma_bg = imuParametersVec_.at(0).sigma_bg;
//      const double sigma_ba = imuParametersVec_.at(0).sigma_ba;
//      std::shared_ptr<ceres::SpeedAndBiasError > speedAndBiasError(
//            new ceres::SpeedAndBiasError(
//                speedAndBias, 1.0, sigma_bg*sigma_bg, sigma_ba*sigma_ba));
//      // add to map
//      mapPtr_->addResidualBlock(
//          speedAndBiasError,
//          NULL,
//          mapPtr_->parameterBlockPtr(
//              states.sensors.at(SensorStates::Imu).at(i).at(ImuSensorStates::SpeedAndBias).id));
//      //mapPtr_->isJacobianCorrect(id,1.0e-6);
//    }
  }
  else{
    // add IMU error terms
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      std::shared_ptr<ceres::ImuError> imuError(
          new ceres::ImuError(imuMeasurements, imuParametersVec_.at(i),
                              lastElementIterator->second.timestamp,
                              states.timestamp));
      /*::ceres::ResidualBlockId id = */mapPtr_->addResidualBlock(
          imuError,
          NULL,
          mapPtr_->parameterBlockPtr(lastElementIterator->second.id),
          mapPtr_->parameterBlockPtr(
              lastElementIterator->second.sensors.at(SensorStates::Imu).at(i).at(
                  ImuSensorStates::SpeedAndBias).id),
          mapPtr_->parameterBlockPtr(states.id),
          mapPtr_->parameterBlockPtr(
              states.sensors.at(SensorStates::Imu).at(i).at(
                  ImuSensorStates::SpeedAndBias).id));
      //imuError->setRecomputeInformation(false);
      //mapPtr_->isJacobianCorrect(id,1.0e-9);
      //imuError->setRecomputeInformation(true);
    }

    // add relative sensor state errors
//    for (size_t i = 0; i < extrinsicsEstimationParametersVec_.size(); ++i) {
//      if(lastElementIterator->second.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id !=
//          states.sensors.at(SensorStates::Camera).at(i).at(CameraSensorStates::T_SCi).id){
//        // i.e. they are different estimated variables, so link them with a temporal error term
//        double dt = (states.timestamp - lastElementIterator->second.timestamp)
//            .toSec();
//        double translationSigmaC = extrinsicsEstimationParametersVec_.at(i)
//            .sigma_c_relative_translation;
//        double translationVariance = translationSigmaC * translationSigmaC * dt;
//        double rotationSigmaC = extrinsicsEstimationParametersVec_.at(i)
//            .sigma_c_relative_orientation;
//        double rotationVariance = rotationSigmaC * rotationSigmaC * dt;
//        std::shared_ptr<ceres::RelativePoseError> relativeExtrinsicsError(
//            new ceres::RelativePoseError(translationVariance,
//                                         rotationVariance));
//        mapPtr_->addResidualBlock(
//            relativeExtrinsicsError,
//            NULL,
//            mapPtr_->parameterBlockPtr(
//                lastElementIterator->second.sensors.at(SensorStates::Camera).at(
//                    i).at(CameraSensorStates::T_SCi).id),
//            mapPtr_->parameterBlockPtr(
//                states.sensors.at(SensorStates::Camera).at(i).at(
//                    CameraSensorStates::T_SCi).id));
//        //mapPtr_->isJacobianCorrect(id,1.0e-6);
//      }
//    }
    // only camera. this is slightly inconsistent, since the IMU error term contains both
    // a term for global states as well as for the sensor-internal ones (i.e. biases).
    // TODO: magnetometer, pressure, ...
  }

  return true;
}

// Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
// The new number of frames in the window will be numKeyframes+numImuFrames.
bool PriorlessEstimator::applyMarginalizationStrategy(
    size_t numKeyframes, size_t numImuFrames,
    okvis::MapPointVector& removedLandmarks)
{
  // keep the newest numImuFrames
  std::map<uint64_t, States>::reverse_iterator rit = statesMap_.rbegin();
  for(size_t k=0; k<numImuFrames; k++){
    rit++;
    if(rit==statesMap_.rend()){
      // nothing to do.
      return true;
    }
  }

  // remove linear marginalizationError, if existing
  if (marginalizationErrorPtr_ && marginalizationResidualId_) {
    bool success = mapPtr_->removeResidualBlock(marginalizationResidualId_);
    OKVIS_ASSERT_TRUE_DBG(Exception, success,
                       "could not remove marginalization error");
    marginalizationResidualId_ = 0;
    if (!success)
      return false;
  }

  // these will keep track of what we want to marginalize out.
  std::vector<uint64_t> paremeterBlocksToBeMarginalized;
  std::vector<bool> keepParameterBlocks;

  if (!marginalizationErrorPtr_) {
    marginalizationErrorPtr_.reset(
        new ceres::MarginalizationError(*mapPtr_.get()));
  }

  // distinguish if we marginalize everything or everything but pose
  std::vector<uint64_t> removeKeyframes;
  std::vector<uint64_t> removeRegularFrames; // pose and replace sensor state factors
  std::vector<uint64_t> removeAllButPose;
  std::vector<uint64_t> allLinearizedFrames;
  size_t countedKeyframes = 0;
  while (rit != statesMap_.rend()) {
    if (!rit->second.isKeyframe) {
      removeRegularFrames.push_back(rit->second.id);
    } else {
      if (countedKeyframes >= numKeyframes) {
        removeKeyframes.push_back(rit->second.id);
        removeAllButPose.push_back(rit->second.id);
      }
      countedKeyframes++;
    }
    allLinearizedFrames.push_back(rit->second.id);
    ++rit;// check the next frame
  }

  // In the case that only the pose of a regular frame will be marginalized
  // without the sensor states, replace its two IMU factors with one IMU factor.
  for (size_t index = 0; index < removeRegularFrames.size(); ++index) {
    std::map<uint64_t, States>::iterator it =
        statesMap_.find(removeRegularFrames[index]);
    // delete the old ImuError
    size_t i = SensorStates::Imu;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = ImuSensorStates::SpeedAndBias;
      OKVIS_ASSERT_TRUE(
          Exception, it->second.sensors[i][j][k].exists,
          "Speed and bias params do not exist for a regular frame!");
      it->second.sensors[i][j][k].exists = false;  // remember we removed
      ceres::Map::ResidualBlockCollection residuals =
          mapPtr_->residuals(it->second.sensors[i][j][k].id);
      OKVIS_ASSERT_EQ(Exception, residuals.size(), 2u,
                      "Except for the very first one, each speed and bias "
                      "param block should have 2 residuals!");
      mapPtr_->removeParameterBlock(it->second.sensors[i][j][k].id);
    }
    mapPtr_->removeParameterBlock(it->second.id);

    // add an alternate ImuError
    // TODO(jhuai): do we cap the time range, e.g., 10 sec as in VINS Mono?
    std::map<uint64_t, States>::iterator previt = it;
    --previt;
    std::map<uint64_t, States>::iterator nextit = it;
    ++nextit;
    okvis::ImuMeasurementDeque imuMeasurements = inertialMeasForStates_.find(
        previt->second.timestamp, nextit->second.timestamp);
    OKVIS_ASSERT_GE(Exception, imuMeasurements.size(), 1, "None imu data found!");
    for (size_t i = 0; i < imuParametersVec_.size(); ++i) {
      std::shared_ptr<ceres::ImuError> imuError(new ceres::ImuError(
          imuMeasurements, imuParametersVec_.at(i), previt->second.timestamp,
          nextit->second.timestamp));
      mapPtr_->addResidualBlock(imuError, NULL,
                                mapPtr_->parameterBlockPtr(previt->second.id),
                                mapPtr_->parameterBlockPtr(
                                    previt->second.sensors.at(SensorStates::Imu)
                                        .at(i)
                                        .at(ImuSensorStates::SpeedAndBias)
                                        .id),
                                mapPtr_->parameterBlockPtr(nextit->second.id),
                                mapPtr_->parameterBlockPtr(
                                    nextit->second.sensors.at(SensorStates::Imu)
                                        .at(i)
                                        .at(ImuSensorStates::SpeedAndBias)
                                        .id));
    }

    multiFramePtrMap_.erase(it->second.id);
    statesMap_.erase(it);
  }

  // marginalize everything but pose:
  for(size_t k = 0; k<removeAllButPose.size(); ++k){
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeAllButPose[k]);
    for (size_t i = 0; i < it->second.global.size(); ++i) {
      if (i == GlobalStates::T_WS) {
        continue; // we do not remove the pose here.
      }
      if (!it->second.global[i].exists) {
        continue; // if it doesn't exist, we don't do anything.
      }
      if (mapPtr_->parameterBlockPtr(it->second.global[i].id)->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if(checkit->second.global[i].exists &&
          checkit->second.global[i].id == it->second.global[i].id){
        continue;
      }
      it->second.global[i].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[i].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
          it->second.global[i].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
            residuals[r].errorInterfacePtr);
        if(!reprojectionError){   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }
    // add all error terms of the sensor states.
    for (size_t i = 0; i < it->second.sensors.size(); ++i) {
      for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
        for (size_t k = 0; k < it->second.sensors[i][j].size(); ++k) {
          if (i == SensorStates::Camera && k == CameraSensorStates::T_SCi) {
            continue; // we do not remove the extrinsics pose here.
          }
          if (!it->second.sensors[i][j][k].exists) {
            continue;
          }
          if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
              ->fixed()) {
            continue;  // we never eliminate fixed blocks.
          }
          std::map<uint64_t, States>::iterator checkit = it;
          checkit++;
          // only get rid of it, if it's different
          if(checkit->second.sensors[i][j][k].exists &&
              checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
            continue;
          }
          it->second.sensors[i][j][k].exists = false; // remember we removed
          paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
          keepParameterBlocks.push_back(false);
          ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
              it->second.sensors[i][j][k].id);
          for (size_t r = 0; r < residuals.size(); ++r) {
            std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
                std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                residuals[r].errorInterfacePtr);
            if(!reprojectionError){   // we make sure no reprojection errors are yet included.
              marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
            }
          }
        }
      }
    }
  }
  // marginalize ONLY pose now:
//  bool reDoFixation = false;
  for(size_t k = 0; k<removeKeyframes.size(); ++k){
    std::map<uint64_t, States>::iterator it = statesMap_.find(removeKeyframes[k]);

    // schedule removal - but always keep the very first frame.
    //if(it != statesMap_.begin()){
    if(true){ /////DEBUG
      it->second.global[GlobalStates::T_WS].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.global[GlobalStates::T_WS].id);
      keepParameterBlocks.push_back(false);
    }

    // add remaing error terms
    ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
        it->second.global[GlobalStates::T_WS].id);

    for (size_t r = 0; r < residuals.size(); ++r) {
//      if(std::dynamic_pointer_cast<ceres::PoseError>(
//           residuals[r].errorInterfacePtr)){ // avoids linearising initial pose error
//				mapPtr_->removeResidualBlock(residuals[r].residualBlockId);
//				reDoFixation = true;
//        continue;
//      }
      std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
          std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
          residuals[r].errorInterfacePtr);
      if(!reprojectionError){   // we make sure no reprojection errors are yet included.
        marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
      }
    }

    // add remaining error terms of the sensor states.
    size_t i = SensorStates::Camera;
    for (size_t j = 0; j < it->second.sensors[i].size(); ++j) {
      size_t k = CameraSensorStates::T_SCi;
      if (!it->second.sensors[i][j][k].exists) {
        continue;
      }
      if (mapPtr_->parameterBlockPtr(it->second.sensors[i][j][k].id)
          ->fixed()) {
        continue;  // we never eliminate fixed blocks.
      }
      std::map<uint64_t, States>::iterator checkit = it;
      checkit++;
      // only get rid of it, if it's different
      if(checkit->second.sensors[i][j][k].exists &&
          checkit->second.sensors[i][j][k].id == it->second.sensors[i][j][k].id){
        continue;
      }
      it->second.sensors[i][j][k].exists = false; // remember we removed
      paremeterBlocksToBeMarginalized.push_back(it->second.sensors[i][j][k].id);
      keepParameterBlocks.push_back(false);
      ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(
          it->second.sensors[i][j][k].id);
      for (size_t r = 0; r < residuals.size(); ++r) {
        std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
            std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
            residuals[r].errorInterfacePtr);
        if(!reprojectionError){   // we make sure no reprojection errors are yet included.
          marginalizationErrorPtr_->addResidualBlock(residuals[r].residualBlockId);
        }
      }
    }

    // now finally we treat all the observations.
    OKVIS_ASSERT_TRUE_DBG(Exception, allLinearizedFrames.size()>0, "bug");
    uint64_t currentKfId = allLinearizedFrames.at(0);

    {
      for(PointMap::iterator pit = landmarksMap_.begin();
          pit != landmarksMap_.end(); ){

        ceres::Map::ResidualBlockCollection residuals = mapPtr_->residuals(pit->first);

        // first check if we can skip
        bool skipLandmark = true;
        bool hasNewObservations = false;
        bool justDelete = false;
        bool marginalize = true;
        bool errorTermAdded = false;
        std::map<uint64_t,bool> visibleInFrame;
        size_t obsCount = 0;
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            // since we have implemented the linearisation to account for robustification,
            // we don't kick out bad measurements here any more like
            // if(vectorContains(allLinearizedFrames,poseId)){ ...
            //   if (error.transpose() * error > 6.0) { ... removeObservation ... }
            // }
            if(vectorContains(removeKeyframes,poseId)){
              skipLandmark = false;
            }
            if(poseId>=currentKfId){
              marginalize = false;
              hasNewObservations = true;
            }
            if(vectorContains(allLinearizedFrames, poseId)){
              visibleInFrame.insert(std::pair<uint64_t,bool>(poseId,true));
              obsCount++;
            }
          }
        }

        if(residuals.size()==0){
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        if(skipLandmark) {
          pit++;
          continue;
        }

        // so, we need to consider it.
        for (size_t r = 0; r < residuals.size(); ++r) {
          std::shared_ptr<ceres::ReprojectionErrorBase> reprojectionError =
              std::dynamic_pointer_cast<ceres::ReprojectionErrorBase>(
                  residuals[r].errorInterfacePtr);
          if (reprojectionError) {
            uint64_t poseId = mapPtr_->parameters(residuals[r].residualBlockId).at(0).first;
            if((vectorContains(removeKeyframes,poseId) && hasNewObservations) ||
                (!vectorContains(allLinearizedFrames,poseId) && marginalize)){
              // ok, let's ignore the observation.
              removeObservation(residuals[r].residualBlockId);
              residuals.erase(residuals.begin() + r);
              r--;
            } else if(marginalize && vectorContains(allLinearizedFrames,poseId)) {
              // TODO: consider only the sensible ones for marginalization
              if(obsCount<2){ //visibleInFrame.size()
                removeObservation(residuals[r].residualBlockId);
                residuals.erase(residuals.begin() + r);
                r--;
              } else {
                // add information to be considered in marginalization later.
                errorTermAdded = true;
                marginalizationErrorPtr_->addResidualBlock(
                    residuals[r].residualBlockId, false);
              }
            }
            // check anything left
            if (residuals.size() == 0) {
              justDelete = true;
              marginalize = false;
            }
          }
        }

        if(justDelete){
          mapPtr_->removeParameterBlock(pit->first);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }
        if(marginalize&&errorTermAdded){
          paremeterBlocksToBeMarginalized.push_back(pit->first);
          keepParameterBlocks.push_back(false);
          removedLandmarks.push_back(pit->second);
          pit = landmarksMap_.erase(pit);
          continue;
        }

        pit++;
      }
    }

    if (it->second.isKeyframe) {
      inertialMeasForStates_.pop_front(it->second.timestamp - half_window_);
    }

    // update book-keeping and go to the next frame
    //if(it != statesMap_.begin()){ // let's remember that we kept the very first pose
    if(true) { ///// DEBUG
      multiFramePtrMap_.erase(it->second.id);
      statesMap_.erase(it->second.id);
    }
  }

  // now apply the actual marginalization
  if(paremeterBlocksToBeMarginalized.size()>0){
    std::vector< ::ceres::ResidualBlockId> addedPriors;
    marginalizationErrorPtr_->marginalizeOut(paremeterBlocksToBeMarginalized, keepParameterBlocks);
  }

  // update error computation
  if(paremeterBlocksToBeMarginalized.size()>0){
    marginalizationErrorPtr_->updateErrorComputation();
  }

  // add the marginalization term again
  if(marginalizationErrorPtr_->num_residuals()==0){
    marginalizationErrorPtr_.reset();
  }
  if (marginalizationErrorPtr_) {
  std::vector<std::shared_ptr<okvis::ceres::ParameterBlock> > parameterBlockPtrs;
  marginalizationErrorPtr_->getParameterBlockPtrs(parameterBlockPtrs);
  marginalizationResidualId_ = mapPtr_->addResidualBlock(
      marginalizationErrorPtr_, NULL, parameterBlockPtrs);
  OKVIS_ASSERT_TRUE_DBG(Exception, marginalizationResidualId_,
                     "could not add marginalization error");
  if (!marginalizationResidualId_)
    return false;
  }
	
//	if(reDoFixation){
//	  // finally fix the first pose properly
//		//mapPtr_->resetParameterization(statesMap_.begin()->first, ceres::Map::Pose3d);
//		okvis::kinematics::Transformation T_WS_0;
//		get_T_WS(statesMap_.begin()->first, T_WS_0);
//	  Eigen::Matrix<double,6,6> information = Eigen::Matrix<double,6,6>::Zero();
//	  information(5,5) = 1.0e14; information(0,0) = 1.0e14; information(1,1) = 1.0e14; information(2,2) = 1.0e14;
//	  std::shared_ptr<ceres::PoseError > poseError(new ceres::PoseError(T_WS_0, information));
//	  mapPtr_->addResidualBlock(poseError,NULL,mapPtr_->parameterBlockPtr(statesMap_.begin()->first));
//	}

  return true;
}

// Start ceres optimization.
#ifdef USE_OPENMP
void PriorlessEstimator::optimize(size_t numIter, size_t numThreads,
                                 bool verbose)
#else
void PriorlessEstimator::optimize(size_t numIter, size_t /*numThreads*/,
                                 bool verbose) // avoid warning since numThreads unused
#warning openmp not detected, your system may be slower than expected
#endif

{
  // assemble options
  mapPtr_->options.linear_solver_type = ::ceres::SPARSE_SCHUR;
  //mapPtr_->options.initial_trust_region_radius = 1.0e4;
  //mapPtr_->options.initial_trust_region_radius = 2.0e6;
  //mapPtr_->options.preconditioner_type = ::ceres::IDENTITY;
  mapPtr_->options.trust_region_strategy_type = ::ceres::DOGLEG;
  //mapPtr_->options.trust_region_strategy_type = ::ceres::LEVENBERG_MARQUARDT;
  //mapPtr_->options.use_nonmonotonic_steps = true;
  //mapPtr_->options.max_consecutive_nonmonotonic_steps = 10;
  //mapPtr_->options.function_tolerance = 1e-12;
  //mapPtr_->options.gradient_tolerance = 1e-12;
  //mapPtr_->options.jacobi_scaling = false;
#ifdef USE_OPENMP
    mapPtr_->options.num_threads = numThreads;
#endif
  mapPtr_->options.max_num_iterations = numIter;

  if (verbose) {
    mapPtr_->options.minimizer_progress_to_stdout = true;
  } else {
    mapPtr_->options.minimizer_progress_to_stdout = false;
  }

  // call solver
  mapPtr_->solve();

  // update landmarks
  {
    for(auto it = landmarksMap_.begin(); it!=landmarksMap_.end(); ++it){
      Eigen::MatrixXd H(3,3);
      mapPtr_->getLhs(it->first,H);
      Eigen::SelfAdjointEigenSolver< Eigen::Matrix3d > saes(H);
      Eigen::Vector3d eigenvalues = saes.eigenvalues();
      const double smallest = (eigenvalues[0]);
      const double largest = (eigenvalues[2]);
      if(smallest<1.0e-12){
        // this means, it has a non-observable depth
        it->second.quality = 0.0;
      } else {
        // OK, well constrained
        it->second.quality = sqrt(smallest)/sqrt(largest);
      }

      // update coordinates
      it->second.pointHomog = std::static_pointer_cast<okvis::ceres::HomogeneousPointParameterBlock>(
          mapPtr_->parameterBlockPtr(it->first))->estimate();
    }
  }

  // summary output
  if (verbose) {
    LOG(INFO) << mapPtr_->summary.FullReport();
  }
}

}  // namespace okvis
