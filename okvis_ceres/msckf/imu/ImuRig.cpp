#include <msckf/imu/ImuRig.hpp>

namespace okvis {
int ImuRig::addImu(const okvis::ImuParameters& imuParams) {
  int modelId = ImuModelNameToId(imuParams.model_type);
  Eigen::Matrix<double, Eigen::Dynamic, 1> euclideanParams;
  Eigen::Quaterniond q_gyro_i = Eigen::Quaterniond::Identity();
  int augmentedEuclideanDim = ImuModelGetAugmentedEuclideanDim(modelId);
  Eigen::Matrix<double, Eigen::Dynamic, 1> nominalAugmentedParams =
      ImuModelNominalAugmentedParams(modelId);
  switch (modelId) {
    case Imu_BG_BA_TG_TS_TA::kModelId:
      euclideanParams.resize(Imu_BG_BA_TG_TS_TA::kGlobalDim, 1);
      euclideanParams.head<3>() = imuParams.g0;
      euclideanParams.segment<3>(3) = imuParams.a0;
      euclideanParams.segment<9>(6) = imuParams.Tg0;
      euclideanParams.segment<9>(15) = imuParams.Ts0;
      euclideanParams.segment<9>(24) = imuParams.Ta0;
      break;
    case ScaledMisalignedImu::kModelId:
      euclideanParams.resize(ScaledMisalignedImu::kGlobalDim - 4, 1);
      euclideanParams.head<3>() = imuParams.g0;
      euclideanParams.segment<3>(3) = imuParams.a0;
      euclideanParams.tail(augmentedEuclideanDim) =
          nominalAugmentedParams.head(augmentedEuclideanDim);
      q_gyro_i.coeffs() = nominalAugmentedParams.tail<4>();
      break;
    default:
      euclideanParams.resize(Imu_BG_BA::kGlobalDim, 1);
      euclideanParams.head<3>() = imuParams.g0;
      euclideanParams.segment<3>(3) = imuParams.a0;
      break;
  }
  imus_.emplace_back(modelId, euclideanParams, q_gyro_i);
  return static_cast<int>(imus_.size()) - 1;
}

void getImuAugmentedStatesEstimate(
    std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>
        imuAugmentedParameterPtrs,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams, int imuModelId) {
  switch (imuModelId) {
    case Imu_BG_BA::kModelId:
      break;
    case Imu_BG_BA_TG_TS_TA::kModelId: {
      extraParams->resize(27, 1);
      std::shared_ptr<const ceres::ShapeMatrixParamBlock> tgParamBlockPtr =
          std::static_pointer_cast<const ceres::ShapeMatrixParamBlock>(
              imuAugmentedParameterPtrs[0]);
      Eigen::Matrix<double, 9, 1> sm = tgParamBlockPtr->estimate();
      extraParams->head<9>() = sm;

      std::shared_ptr<const ceres::ShapeMatrixParamBlock> tsParamBlockPtr =
          std::static_pointer_cast<const ceres::ShapeMatrixParamBlock>(
              imuAugmentedParameterPtrs[1]);
      sm = tsParamBlockPtr->estimate();
      extraParams->segment<9>(9) = sm;

      std::shared_ptr<const ceres::ShapeMatrixParamBlock> taParamBlockPtr =
          std::static_pointer_cast<const ceres::ShapeMatrixParamBlock>(
              imuAugmentedParameterPtrs[2]);
      sm = taParamBlockPtr->estimate();
      extraParams->segment<9>(18) = sm;
    } break;
    case ScaledMisalignedImu::kModelId:
      LOG(WARNING) << "get state estimate not implemented for imu model "
                   << imuModelId;
      break;
  }
}
}  // namespace okvis
