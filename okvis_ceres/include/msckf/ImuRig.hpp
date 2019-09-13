#ifndef INCLUDE_MSCKF_IMU_RIG_HPP_
#define INCLUDE_MSCKF_IMU_RIG_HPP_

#include <okvis/kinematics/Transformation.hpp>

namespace okvis {

// TODO(jhuai): put the enum members as kModelId into each IMU model
enum ImuModelType {
  BG_BA = 0,
  BG_BA_TG_TS_TA = 1,
  BG_BA_SMRG_SMA = 2,
};

class ImuRig {
  std::vector<int> model_type_;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline ImuRig() {}

  inline int getModelType(int index) {
    if ((int)model_type_.size() > index) {
      return model_type_[index];
    } else {
      return -1;
    }
  }

  inline int addImu() {
    model_type_.push_back(BG_BA);
    return static_cast<int>(model_type_.size()) - 1;
  }

  inline int getMinimalImuParamDimen(int imu_id=0) {
      switch (model_type_[imu_id]) {
      case BG_BA:
          return 6;
      case BG_BA_TG_TS_TA:
          return 6 + 27;
      case BG_BA_SMRG_SMA:
          return 6 + 9 + 6;
      default:
          return 6;
      }
  }
};

inline ImuModelType ImuModelNameToId(std::string imu_model) {
  std::transform(imu_model.begin(), imu_model.end(),
                 imu_model.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (imu_model.compare("BG_BA") == 0) {
    return BG_BA;
  } else if (imu_model.compare("BG_BA_TG_TS_TA") == 0) {
    return BG_BA_TG_TS_TA;
  } else if (imu_model.compare("BG_BA_SMRG_SMA") == 0) {
    return BG_BA_SMRG_SMA;
  } else {
    return BG_BA;
  }
}

inline void ImuModelToFormatString(const ImuModelType imu_model,
                                const std::string delimiter,
                                std::string* format_string) {
  std::stringstream stream;
  stream << "b_g_x[rad/s]" << delimiter << "b_g_y" << delimiter << "b_g_z"
         << delimiter << "b_a_x[m/s^2]" << delimiter << "b_a_y" << delimiter
         << "b_a_z";
  switch (imu_model) {
    case BG_BA_TG_TS_TA:
      stream << delimiter << "Tg_1" << delimiter << "Tg_2" << delimiter
             << "Tg_3" << delimiter << "Tg_4" << delimiter << "Tg_5"
             << delimiter << "Tg_6" << delimiter << "Tg_7" << delimiter
             << "Tg_8" << delimiter << "Tg_9" << delimiter << "Ts_1"
             << delimiter << "Ts_2" << delimiter << "Ts_3" << delimiter
             << "Ts_4" << delimiter << "Ts_5" << delimiter << "Ts_6"
             << delimiter << "Ts_7" << delimiter << "Ts_8" << delimiter
             << "Ts_9" << delimiter << "Ta_1" << delimiter << "Ta_2"
             << delimiter << "Ta_3" << delimiter << "Ta_4" << delimiter
             << "Ta_5" << delimiter << "Ta_6" << delimiter << "Ta_7"
             << delimiter << "Ta_8" << delimiter << "Ta_9";
      break;
    case BG_BA_SMRG_SMA:
      stream << delimiter << "Sg_x" << delimiter << "Sg_y" << delimiter
             << "Sg_z" << delimiter << "Mg_xy" << delimiter << "Mg_xz"
             << delimiter << "Mg_yz" << delimiter << "Rg_x" << delimiter
             << "Rg_y" << delimiter << "Rg_z" << delimiter << "Rg_w"
             << delimiter << "Sa_x" << delimiter << "Sa_y" << delimiter
             << "Sa_z" << delimiter << "Ma_xy" << delimiter << "Ma_xz"
             << delimiter << "Ma_yz";
      break;
    case BG_BA:
    default:
      break;
  }
  *format_string = stream.str();
}

}  // namespace okvis
#endif  // INCLUDE_MSCKF_IMU_RIG_HPP_
