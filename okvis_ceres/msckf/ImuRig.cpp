#include <msckf/ImuRig.hpp>

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
}  // namespace okvis
