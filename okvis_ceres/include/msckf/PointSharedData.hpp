#ifndef INCLUDE_MSCKF_POINT_SHARED_DATA_HPP_
#define INCLUDE_MSCKF_POINT_SHARED_DATA_HPP_

#include <memory>
#include <unordered_map>
#include <Eigen/StdVector>

#include <msckf/RemoveFromVector.hpp>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>

namespace msckf {
// The state info for one keypoint relevant to computing the pose (T_WB) and
// (linear and angular) velocity (v_WB, omega_WB_B) at keypoint observation epoch.
struct StateInfoForOneKeypoint {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StateInfoForOneKeypoint() {

    }
    StateInfoForOneKeypoint(
        uint64_t _frameId, size_t _camIdx,
        std::shared_ptr<const okvis::ceres::ParameterBlock> T_WB_ptr,
        double _normalizedRow)
        : frameId(_frameId),
          cameraId(_camIdx),
          T_WBj_ptr(T_WB_ptr),
          normalizedRow(_normalizedRow) {}

    uint64_t frameId;
    size_t cameraId;
    std::shared_ptr<const okvis::ceres::ParameterBlock> T_WBj_ptr;
    std::shared_ptr<const okvis::ceres::ParameterBlock> speedAndBiasPtr;
    // IMU measurements covering the state epoch.
    std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurementPtr;
    okvis::Time stateEpoch;
    double tdAtCreation;
    double normalizedRow; // v / imageHeight - 0.5.
    // linearization points at the state.
    std::shared_ptr<const Eigen::Matrix<double, 6, 1>> positionVelocityPtr;
    // Pose of the body frame in the world frame at the feature observation epoch.
    // It should be computed with IMU propagation for RS cameras.
    okvis::kinematics::Transformation T_WBtij;
    Eigen::Vector3d v_WBtij;
    Eigen::Vector3d omega_Btij;
    okvis::kinematics::Transformation  lP_T_WBtij;
    Eigen::Vector3d lP_v_WBtij;
};

enum class PointSharedDataState {
  Barebones = 0,
  ImuInfoReady = 1,
  NavStateReady = 2,
  NavStateForJacReady = 3,
};

// Data shared by observations of a point landmark in computing Jacobians
// relative to pose (T_WB) and velocity (v_WB) and camera time parameters.
// The data of the class members may be updated in ceres EvaluationCallback.
class PointSharedData {
 public:

  typedef std::vector<StateInfoForOneKeypoint,
                      Eigen::aligned_allocator<StateInfoForOneKeypoint>>
      StateInfoForObservationsType;

  PointSharedData() : status_(PointSharedDataState::Barebones) {}

  // assume the observations are in the decreasing order of state age.
  void addKeypointObservation(
      const okvis::KeypointIdentifier& kpi,
      std::shared_ptr<const okvis::ceres::ParameterBlock> T_WBj_ptr,
      double normalizedRow) {
    stateInfoForObservations_.emplace_back(kpi.frameId, kpi.cameraIndex,
                                           T_WBj_ptr, normalizedRow);
  }

  /// @name Setters for data for IMU propagation.
  /// @{
  void setVelocityParameterBlockPtr(
      int index,
      std::shared_ptr<const okvis::ceres::ParameterBlock> speedAndBiasPtr) {
    stateInfoForObservations_[index].speedAndBiasPtr = speedAndBiasPtr;
  }

  void setImuInfo(
      int index, const okvis::Time stateEpoch, double td0,
      std::shared_ptr<const okvis::ImuMeasurementDeque> imuMeasurements,
      std::shared_ptr<const Eigen::Matrix<double, 6, 1>> positionVelocityPtr) {
    stateInfoForObservations_[index].stateEpoch = stateEpoch;
    stateInfoForObservations_[index].tdAtCreation = td0;
    stateInfoForObservations_[index].imuMeasurementPtr = imuMeasurements;
    stateInfoForObservations_[index].positionVelocityPtr = positionVelocityPtr;
  }

  void setImuAugmentedParameterPtrs(
      const std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>&
          imuAugmentedParamBlockPtrs,
      const okvis::ImuParameters* imuParams) {
    imuAugmentedParamBlockPtrs_ = imuAugmentedParamBlockPtrs;
    imuParameters_ = imuParams;
  }

  void setCameraTimeParameterPtrs(
      const std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>&
          tdParamBlockPtrs,
      const std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>>&
          trParamBlockPtrs) {
    tdParamBlockPtrs_ = tdParamBlockPtrs;
    trParamBlockPtrs_ = trParamBlockPtrs;
    status_ = PointSharedDataState::ImuInfoReady;
  }
  /// @}

  /// @name functions for IMU propagation.
  /// @{
  /**
   * @brief computePoseAndVelocityAtObservation.
   *     for feature i, estimate $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$,
   *     $v_B^G(t_{f_i})$, and $\omega_{GB}^B(t_{f_i})$ with the corresponding
   *     states' LATEST ESTIMATES and imu measurements.
   * @warning Call this function after setImuAugmentedParameterPtrs().
   */
  void computePoseAndVelocityAtObservation();

  /**
   * @brief computePoseAndVelocityForJacobians
   * @warning Only call this function after
   * computePoseAndVelocityAtObservation() has finished.
   * @param useLinearizationPoint
   */
  void computePoseAndVelocityForJacobians(bool useLinearizationPoint);
  /// @}

  void computeSharedJacobians(int cameraObservationModelId);

  /// @name Functions for anchors.
  /// @{
  void setAnchors(const std::vector<okvis::AnchorFrameIdentifier>& anchorIds) {
    anchorIds_ = anchorIds;    
  }

  const std::vector<okvis::AnchorFrameIdentifier>& anchorIds() const {
    return anchorIds_;
  }

  /**
   * @brief Get index of observations from the anchor frames in the observation sequence.
   * @warning This only support a monocular camera.
   * @return
   */
  std::vector<int> anchorObservationIds() const;

  okvis::kinematics::Transformation T_WB_mainAnchor() const {
    return stateInfoForObservations_[anchorIds_[0].observationIndex_].T_WBtij;
  }

  okvis::kinematics::Transformation T_WB_mainAnchorForJacobian() const {
    return stateInfoForObservations_[anchorIds_[0].observationIndex_].lP_T_WBtij;
  }

  okvis::kinematics::Transformation T_WB_mainAnchorStateEpoch() const {
    const StateInfoForOneKeypoint& mainAnchorItem =
        stateInfoForObservations_.at(anchorIds_[0].observationIndex_);
    return std::static_pointer_cast<const okvis::ceres::PoseParameterBlock>(
            mainAnchorItem.T_WBj_ptr)
            ->estimate();
  }

  okvis::kinematics::Transformation T_WB_mainAnchorStateEpochForJacobian() const {
    const StateInfoForOneKeypoint& mainAnchorItem =
        stateInfoForObservations_.at(anchorIds_[0].observationIndex_);
    okvis::kinematics::Transformation lP_T_WB =
        std::static_pointer_cast<const okvis::ceres::PoseParameterBlock>(
            mainAnchorItem.T_WBj_ptr)
            ->estimate();
    std::shared_ptr<const Eigen::Matrix<double, 6, 1>> posVelFirstEstimatePtr =
        mainAnchorItem.positionVelocityPtr;
    lP_T_WB = okvis::kinematics::Transformation(
        posVelFirstEstimatePtr->head<3>(), lP_T_WB.q());
    return lP_T_WB;
  }
  /// @}

  /// @name functions for managing the main stateInfo list.
  /// @{
  StateInfoForObservationsType::iterator begin() {
    return stateInfoForObservations_.begin();
  }

  StateInfoForObservationsType::iterator end() {
      return stateInfoForObservations_.end();
  }

  void removeBadObservations(const std::vector<bool>& projectStatus) {
      removeUnsetMatrices<StateInfoForOneKeypoint>(&stateInfoForObservations_, projectStatus);
  }

  /**
   * @brief removeExtraObservations
   * @warning orderedSelectedFrameIds must be a subsets of stateInfoForObservations_
   * @param orderedSelectedFrameIds
   * @param imageNoise2dStdList
   */
  void removeExtraObservations(const std::vector<uint64_t>& orderedSelectedFrameIds,
                               std::vector<double>* imageNoise2dStdList);

  void removeExtraObservationsLegacy(
      const std::vector<uint64_t>& orderedSelectedFrameIds,
      std::vector<double>* imageNoise2dStdList);
  /// @}

  /// @name Getters for frameIds.
  /// @{
  size_t numObservations() const {
      return stateInfoForObservations_.size();
  }

  std::vector<std::pair<uint64_t, size_t>> frameIds() const {
    std::vector<std::pair<uint64_t, size_t>> frameIds;
    frameIds.reserve(stateInfoForObservations_.size());
    for (auto item : stateInfoForObservations_) {
      frameIds.emplace_back(item.frameId, item.cameraId);
    }
    return frameIds;
  }

  uint64_t frameId(int index) const {
    return stateInfoForObservations_[index].frameId;
  }

  uint64_t lastFrameId() const {
    return stateInfoForObservations_.back().frameId;
  }
  /// @}

  /// @name Getters
  /// @{
  double normalizedFeatureTime(int observationIndex) const {
    size_t cameraIdx = stateInfoForObservations_[observationIndex].cameraId;
    return tdParamBlockPtrs_[cameraIdx]->parameters()[0] +
           trParamBlockPtrs_[cameraIdx]->parameters()[0] *
               stateInfoForObservations_[observationIndex].normalizedRow -
           stateInfoForObservations_[observationIndex].tdAtCreation;
  }

  size_t cameraIndex(size_t observationIndex) const {
    return stateInfoForObservations_[observationIndex].cameraId;
  }

  double normalizedRow(int index) const {
    return stateInfoForObservations_[index].normalizedRow;
  }

  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
  poseAtObservationList() const {
    std::vector<okvis::kinematics::Transformation,
                Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        T_WBtij_list;
    T_WBtij_list.reserve(stateInfoForObservations_.size());
    for (auto item : stateInfoForObservations_) {
      T_WBtij_list.push_back(item.T_WBtij);
    }
    return T_WBtij_list;
  }

  std::vector<size_t> cameraIndexList() const {
    std::vector<size_t> camIndices;
    camIndices.reserve(stateInfoForObservations_.size());
    for (auto item : stateInfoForObservations_) {
      camIndices.push_back(item.cameraId);
    }
    return camIndices;
  }

  void poseAtObservation(int index, okvis::kinematics::Transformation* T_WBtij) const {
    *T_WBtij = stateInfoForObservations_[index].T_WBtij;
  }

  okvis::kinematics::Transformation T_WBtij(int index) const {
    return stateInfoForObservations_[index].T_WBtij;
  }

  Eigen::Vector3d omega_Btij(int index) const {
    return stateInfoForObservations_[index].omega_Btij;
  }

  okvis::kinematics::Transformation T_WBtij_ForJacobian(int index) const {
    return stateInfoForObservations_[index].lP_T_WBtij;
  }

  Eigen::Vector3d v_WBtij_ForJacobian(int index) const {
    return stateInfoForObservations_[index].lP_v_WBtij;
  }

  PointSharedDataState status() const {
    return status_;
  }

  double gravityNorm() const {
    return imuParameters_->g;
  }
  /// @}

  /// @name Getters for parameter blocks
  /// @{
  std::shared_ptr<const okvis::ceres::PoseParameterBlock> poseParameterBlockPtr(
      int observationIndex) const {
    return std::static_pointer_cast<const okvis::ceres::PoseParameterBlock>(
        stateInfoForObservations_.at(observationIndex).T_WBj_ptr);
  }

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  speedAndBiasParameterBlockPtr(int observationIndex) const {
    return stateInfoForObservations_.at(observationIndex).speedAndBiasPtr;
  }

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  cameraTimeDelayParameterBlockPtr(size_t cameraIndex) const {
    return tdParamBlockPtrs_[cameraIndex];
  }

  std::shared_ptr<const okvis::ceres::ParameterBlock>
  frameReadoutTimeParameterBlockPtr(size_t cameraIndex) const {
    return trParamBlockPtrs_[cameraIndex];
  }
  /// @}

 private:
  // The items of stateInfoForObservations_ are added in an ordered manner
  // by sequentially examining the ordered elements of MapPoint.observations.
  std::vector<StateInfoForOneKeypoint,
              Eigen::aligned_allocator<StateInfoForOneKeypoint>>
      stateInfoForObservations_;

  std::vector<okvis::AnchorFrameIdentifier> anchorIds_;
  std::vector<okvis::kinematics::Transformation,
              Eigen::aligned_allocator<okvis::kinematics::Transformation>>
      T_WBa_list_;

  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> tdParamBlockPtrs_;
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> trParamBlockPtrs_;
  std::vector<std::shared_ptr<const okvis::ceres::ParameterBlock>> imuAugmentedParamBlockPtrs_;
  const okvis::ImuParameters* imuParameters_;

  // The structure of sharedJacobians is determined by an external cameraObservationModelId.
  std::vector<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>,
      Eigen::aligned_allocator<Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>> sharedJacobians_;
  PointSharedDataState status_;
};
} // namespace msckf

#endif // INCLUDE_MSCKF_POINT_SHARED_DATA_HPP_
