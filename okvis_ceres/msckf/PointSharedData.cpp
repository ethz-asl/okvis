#include "msckf/PointSharedData.hpp"
#include <msckf/ImuRig.hpp>
#include <msckf/ImuOdometry.h>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>

namespace msckf {
void PointSharedData::computePoseAndVelocityAtObservation() {
  int imuModelId = okvis::ImuModelNameToId(imuParameters_->model_type);
  Eigen::Matrix<double, -1, 1> vTGTSTA;
  okvis::getImuAugmentedStatesEstimate(
      imuAugmentedParamBlockPtrs_, &vTGTSTA,
      imuModelId);
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
      IMUErrorModel<double> iem(sbj.tail<6>(), vTGTSTA, true);
      okvis::ImuMeasurement interpolatedInertialData;
      okvis::IMUOdometry::interpolateInertialData(*item.imuMeasurementPtr, iem,
                                                  item.stateEpoch,
                                                  interpolatedInertialData);
      item.omega_Btij = interpolatedInertialData.measurement.gyroscopes;
    }
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
    okvis::Duration featureTime(tdParamBlockPtr_->parameters()[0] +
                                trParamBlockPtr_->parameters()[0] *
                                    item.normalizedRow -
                                item.tdAtCreation);

    okvis::ImuMeasurement interpolatedInertialData;
    okvis::poseAndVelocityAtObservation(*item.imuMeasurementPtr, vTGTSTA.data(),
                                        *imuParameters_, item.stateEpoch,
                                        featureTime, &T_WB, &sb,
                                        &interpolatedInertialData, false);
    item.T_WBtij = T_WB;
    item.v_WBtij = sb.head<3>();
    item.omega_Btij = interpolatedInertialData.measurement.gyroscopes;
  }
}

void PointSharedData::computePoseAndVelocityForJacobians(
    bool useLinearizationPoint) {
  if (useLinearizationPoint) {
    Eigen::Matrix<double, -1, 1> vTGTSTA;
    okvis::getImuAugmentedStatesEstimate(
        imuAugmentedParamBlockPtrs_, &vTGTSTA,
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
      okvis::Duration featureTime(tdParamBlockPtr_->parameters()[0] +
                                  trParamBlockPtr_->parameters()[0] *
                                      item.normalizedRow -
                                  item.tdAtCreation);
      okvis::poseAndLinearVelocityAtObservation(
          *item.imuMeasurementPtr, vTGTSTA.data(), *imuParameters_,
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
}

void PointSharedData::computeSharedJacobians() {}
}
