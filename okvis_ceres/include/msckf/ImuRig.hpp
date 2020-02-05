#ifndef INCLUDE_MSCKF_IMU_RIG_HPP_
#define INCLUDE_MSCKF_IMU_RIG_HPP_

#include <memory>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <msckf/ImuModels.hpp>
#include <msckf/EuclideanParamBlockSized.hpp>

#include <okvis/Parameters.hpp>

namespace okvis {

class ImuModel {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  /**
   * @brief ImuModel
   * @param modelId
   * @param euclideanParams
   * @param q_gyro_i quaternion from the accelerometer triad frame to the gyro triad frame
   */
  ImuModel(int modelId, const Eigen::VectorXd& euclideanParams, const Eigen::Quaterniond& q_gyro_i) :
    modelId_(modelId), euclideanParams_(euclideanParams), rotationParams_(q_gyro_i) {

  }

  inline Eigen::VectorXd getImuAugmentedEuclideanParams() const {
    return euclideanParams_.tail(ImuModelGetAugmentedEuclideanDim(modelId_));
  }

  inline void correct() {

  }

  inline int augmentedParamDim() const {
      return 0;
  }

  inline int paramDim() const {
      return 0;
  }

  inline int modelId() const {
    return modelId_;
  }

  Eigen::VectorXd computeImuAugmentedParamsError() const {
    return ImuModelComputeAugmentedParamsError(modelId_, euclideanParams_, rotationParams_);
  }

  inline void setImuAugmentedEuclideanParams(const Eigen::VectorXd& euclideanParams) {
    euclideanParams_.tail(ImuModelGetAugmentedEuclideanDim(modelId_)) = euclideanParams;
  }

private:
  int modelId_;
  Eigen::VectorXd euclideanParams_; // bg ba and extra Euclidean params
  Eigen::Quaterniond rotationParams_; // usually q_gyro_accelerometer
};

class ImuRig {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  inline ImuRig() {}
  inline int getModelId(int index) const {
    if ((int)imus_.size() > index) {
      return imus_[index].modelId();
    } else {
      return -1;
    }
  }

  int addImu(const okvis::ImuParameters& imuParams);

  inline int getImuParamsMinimalDim(int imu_id=0) const {
    return ImuModelGetMinimalDim(imus_[imu_id].modelId());
  }

  inline Eigen::VectorXd getImuAugmentedEuclideanParams(int imu_id=0) const {
    return imus_.at(imu_id).getImuAugmentedEuclideanParams();
  }

  inline Eigen::VectorXd computeImuAugmentedParamsError(int imu_id=0) const {
    return imus_.at(imu_id).computeImuAugmentedParamsError();
  }

  inline void setImuAugmentedEuclideanParams(int imu_id, const Eigen::VectorXd& euclideanParams) {
    imus_.at(imu_id).setImuAugmentedEuclideanParams(euclideanParams);
  }

  inline int getAugmentedMinimalDim(int imu_id) const {
    return ImuModelGetAugmentedMinimalDim(imus_[imu_id].modelId());
  }

  inline int getAugmentedDim(int imu_id) const {
    return ImuModelGetAugmentedDim(imus_[imu_id].modelId());
  }
private:
  std::vector<ImuModel> imus_;
};

/**
 * @brief getImuAugmentedStatesEstimate get augmented IMU parameters except for
 *     biases from parameter blocks.
 * @param imuAugmentedParameterPtrs[in]
 * @param extraParams[out]
 * @param imuModelId[in]
 */
void getImuAugmentedStatesEstimate(
     std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> imuAugmentedParameterPtrs,
    Eigen::Matrix<double, Eigen::Dynamic, 1>* extraParams, int imuModelId);

}  // namespace okvis
#endif  // INCLUDE_MSCKF_IMU_RIG_HPP_
