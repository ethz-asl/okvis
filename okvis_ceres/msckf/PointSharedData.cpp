#include "msckf/PointSharedData.hpp"
#include <iterator>
#include <glog/logging.h>

#include <msckf/CameraRig.hpp>
#include <msckf/imu/ImuRig.hpp>
#include <msckf/imu/ImuOdometry.h>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

namespace msckf {
void PointSharedData::computePoseAndVelocityAtObservation() {
  CHECK(status_ >= PointSharedDataState::ImuInfoReady)
      << "Set IMU data, params, camera time params before calling this method.";
  int imuModelId = okvis::ImuModelNameToId(imuParameters_->model_type);
  Eigen::Matrix<double, -1, 1> imuAugmentedParams;
  okvis::getImuAugmentedStatesEstimate(imuAugmentedParamBlockPtrs_,
                                       &imuAugmentedParams, imuModelId);
  if (0) {
    // naive approach, ignoring the rolling shutter effect and the time offset.
    for (auto& item : stateInfoForObservations_) {
      std::shared_ptr<const okvis::ceres::ParameterBlock> b = item.T_WBj_ptr;
      item.T_WBtij =
          std::static_pointer_cast<const okvis::ceres::PoseParameterBlock>(b)
              ->estimate();
      okvis::SpeedAndBiases sbj =
          std::static_pointer_cast<
              const okvis::ceres::SpeedAndBiasParameterBlock>(
              item.speedAndBiasPtr)
              ->estimate();
      item.v_WBtij = sbj.head<3>();
      ImuErrorModel<double> iem(sbj.tail<6>(), imuAugmentedParams, true);
      okvis::ImuMeasurement interpolatedInertialData;
      okvis::ImuOdometry::interpolateInertialData(*item.imuMeasurementPtr, iem,
                                                  item.stateEpoch,
                                                  interpolatedInertialData);
      item.omega_Btij = interpolatedInertialData.measurement.gyroscopes;
    }
    status_ = PointSharedDataState::NavStateReady;
    return;
  }
  for (auto& item : stateInfoForObservations_) {
    okvis::kinematics::Transformation T_WB =
        std::static_pointer_cast<const okvis::ceres::PoseParameterBlock>(
            item.T_WBj_ptr)
            ->estimate();
    okvis::SpeedAndBiases sb =
        std::static_pointer_cast<
            const okvis::ceres::SpeedAndBiasParameterBlock>(
            item.speedAndBiasPtr)
            ->estimate();
    okvis::Duration featureTime(normalizedFeatureTime(item));
    okvis::ImuMeasurement interpolatedInertialData;
    okvis::poseAndVelocityAtObservation(*item.imuMeasurementPtr, imuAugmentedParams,
                                        *imuParameters_, item.stateEpoch,
                                        featureTime, &T_WB, &sb,
                                        &interpolatedInertialData, false);
    item.T_WBtij = T_WB;
    item.v_WBtij = sb.head<3>();
    item.omega_Btij = interpolatedInertialData.measurement.gyroscopes;
  }
  status_ = PointSharedDataState::NavStateReady;
}

void PointSharedData::computePoseAndVelocityForJacobians(
    bool useFirstEstimate) {
  CHECK(status_ == PointSharedDataState::NavStateReady);
  if (useFirstEstimate) {
    Eigen::Matrix<double, -1, 1> imuAugmentedParams;
    okvis::getImuAugmentedStatesEstimate(
        imuAugmentedParamBlockPtrs_, &imuAugmentedParams,
        okvis::ImuModelNameToId(imuParameters_->model_type));
    for (auto& item : stateInfoForObservations_) {
      okvis::kinematics::Transformation lP_T_WB =
          std::static_pointer_cast<const okvis::ceres::PoseParameterBlock>(
              item.T_WBj_ptr)
              ->estimate();
      okvis::SpeedAndBiases lP_sb =
          std::static_pointer_cast<
              const okvis::ceres::SpeedAndBiasParameterBlock>(
              item.speedAndBiasPtr)
              ->estimate();
      std::shared_ptr<const Eigen::Matrix<double, 6, 1>>
          posVelFirstEstimatePtr = item.positionVelocityPtr;
      lP_T_WB = okvis::kinematics::Transformation(
          posVelFirstEstimatePtr->head<3>(), lP_T_WB.q());
      lP_sb.head<3>() = posVelFirstEstimatePtr->tail<3>();
      okvis::Duration featureTime(normalizedFeatureTime(item));
      okvis::poseAndLinearVelocityAtObservation(
          *item.imuMeasurementPtr, imuAugmentedParams, *imuParameters_,
          item.stateEpoch, featureTime, &lP_T_WB, &lP_sb);
      item.lP_v_WBtij = lP_sb.head<3>();
      item.lP_T_WBtij = lP_T_WB;
    }
  } else {
    for (auto& item : stateInfoForObservations_) {
      item.lP_T_WBtij = item.T_WBtij;
      item.lP_v_WBtij = item.v_WBtij;
    }
  }
  status_ = PointSharedDataState::NavStateForJacReady;
}

void PointSharedData::computeSharedJacobians(int cameraObservationModelId) {
  CHECK(status_ == PointSharedDataState::NavStateForJacReady);
  if (cameraObservationModelId == okvis::cameras::kChordalDistanceId) {

  }
}

void PointSharedData::removeExtraObservations(
    const std::vector<uint64_t>& orderedSelectedFrameIds,
    std::vector<double>* imageNoise2dStdList) {
  CHECK_EQ(imageNoise2dStdList->size(), 2u * stateInfoForObservations_.size());
  auto stateIter = stateInfoForObservations_.begin();
  auto keepStateIter = stateInfoForObservations_.begin();
  auto selectedFrameIter = orderedSelectedFrameIds.begin();
  auto noiseIter = imageNoise2dStdList->begin();
  auto keepNoiseIter = imageNoise2dStdList->begin();

  for (; selectedFrameIter != orderedSelectedFrameIds.end();
       ++selectedFrameIter) {
    while (stateIter->frameId != *selectedFrameIter) {
      ++stateIter;
      noiseIter += 2;
    }
    *keepStateIter = *stateIter;
    ++stateIter;
    ++keepStateIter;

    *keepNoiseIter = *noiseIter;
    ++keepNoiseIter;
    ++noiseIter;
    *keepNoiseIter = *noiseIter;
    ++keepNoiseIter;
    ++noiseIter;
  }
  size_t keepSize = orderedSelectedFrameIds.size();
  size_t foundSize =
      std::distance(stateInfoForObservations_.begin(), keepStateIter);
  CHECK_EQ(orderedSelectedFrameIds.size(), foundSize);
  stateInfoForObservations_.resize(keepSize);
  imageNoise2dStdList->resize(keepSize * 2);

  // Also update the observation index for anchor frames.
  for (std::vector<okvis::AnchorFrameIdentifier>::iterator anchorIdIter =
           anchorIds_.begin();
       anchorIdIter != anchorIds_.end(); ++anchorIdIter) {
    bool found = false;
    int index = (int)stateInfoForObservations_.size() - 1;
    for (auto riter = stateInfoForObservations_.rbegin();
         riter != stateInfoForObservations_.rend(); ++riter, --index) {
      if (riter->frameId == anchorIdIter->frameId_ &&
          riter->cameraId == anchorIdIter->cameraIndex_) {
        anchorIdIter->observationIndex_ = index;
        found = true;
        break;
      }
    }
    LOG_IF(WARNING, !found)
        << "Observation for anchor frame is not found in stateInfo list!";
  }
}

void PointSharedData::removeExtraObservationsLegacy(
    const std::vector<uint64_t>& orderedSelectedFrameIds,
    std::vector<double>* imageNoise2dStdList) {
  auto itFrameIds = stateInfoForObservations_.begin();
  auto itRoi = imageNoise2dStdList->begin();
  size_t numPoses = stateInfoForObservations_.size();
  for (size_t kale = 0; kale < numPoses; ++kale) {
    uint64_t poseId = itFrameIds->frameId;
    if (std::find(orderedSelectedFrameIds.begin(),
                  orderedSelectedFrameIds.end(),
                  poseId) == orderedSelectedFrameIds.end()) {
      itFrameIds = stateInfoForObservations_.erase(itFrameIds);
      itRoi = imageNoise2dStdList->erase(itRoi);
      itRoi = imageNoise2dStdList->erase(itRoi);
      continue;
    } else {
      ++itFrameIds;
      ++itRoi;
      ++itRoi;
    }
  }
}

std::vector<int> PointSharedData::anchorObservationIds() const {
  std::vector<int> anchorObservationIds;
  anchorObservationIds.reserve(2);
  for (auto identifier : anchorIds_) {
    int index = 0;
    for (auto& stateInfo : stateInfoForObservations_) {
      if (identifier.frameId_ == stateInfo.frameId) {
        anchorObservationIds.push_back(index);
        break;
      }
      ++index;
    }
  }
  return anchorObservationIds;
}
} // namespace msckf
