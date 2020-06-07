#include <msckf/checkSensorRig.hpp>

namespace okvis {
bool doesExtrinsicModelFitImuModel(std::string extrinsicModel,
                                   std::string imuModel) {
  int extrinsicModelId = okvis::ExtrinsicModelNameToId(extrinsicModel, nullptr);
  int imuModelId = okvis::ImuModelNameToId(imuModel);
  switch (imuModelId) {
    case Imu_BG_BA_TG_TS_TA::kModelId:
      if (extrinsicModelId != Extrinsic_p_CB::kModelId) {
        LOG(WARNING) << "When IMU model is BG_BA_TG_TS_TA, the first camera's "
                        "extrinsic model should be P_CB!";
        return false;
      }
      break;
    case Imu_BG_BA::kModelId:
    case ScaledMisalignedImu::kModelId:
      if (extrinsicModelId != Extrinsic_p_BC_q_BC::kModelId) {
        LOG(WARNING) << "When IMU model is BG_BA or ScaledMisalignedImu, the "
                        "first camera's extrinsic model should be P_BC_Q_BC!";
        return false;
      }
      break;
    default:
      break;
  }
  return true;
}

}  // namespace okvis
