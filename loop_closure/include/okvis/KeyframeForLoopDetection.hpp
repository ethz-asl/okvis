#ifndef INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#define INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <okvis/MultiFrame.hpp>
#include <okvis/class_macros.hpp>

#include <okvis/InverseTransformMultiplyJacobian.hpp>

namespace okvis {
enum class PoseConstraintType {
  Odometry = 0,
  LoopClosure = 1,
};

class NeighborConstraintInDatabase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeighborConstraintInDatabase();
  NeighborConstraintInDatabase(uint64_t id, okvis::Time stamp,
                               const okvis::kinematics::Transformation& T_BnBr,
                               PoseConstraintType type);
  ~NeighborConstraintInDatabase();

  uint64_t id_;
  okvis::Time stamp_;

  // Br is a body frame for reference, B body frame of this neighbor.
  okvis::kinematics::Transformation T_BBr_;

  PoseConstraintType type_;

  // square root info L' of the inverse of the covariance of the between factor
  // unwhitened/raw error due to measurement noise. LL' = \Lambda = inv(cov)
  // It depends on definitions of the between factor and errors of the
  // measurement. e.g., gtsam::BetweenFactor<Pose3>==log(T_z^{-1}T_x^{-1}T_y)
  // and error of T_z is defined by T_z = Pose3::Retraction(\hat{T}_z, \delta).
  Eigen::Matrix<double, 6, 6> squareRootInfo_;
};

class NeighborConstraintMessage {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeighborConstraintMessage();
  /**
   * @brief NeighborConstraintMessage
   * @param id
   * @param stamp
   * @param T_BnBr Bn the body frame associated with this neighbor,
   * Br the body frame associated with the reference frame of this neighbor.
   * @param T_WB pose of this neighbor.
   * @param type
   */
  NeighborConstraintMessage(
      uint64_t id, okvis::Time stamp,
      const okvis::kinematics::Transformation& T_BnBr,
      const okvis::kinematics::Transformation& T_WB,
      PoseConstraintType type = PoseConstraintType::Odometry);
  ~NeighborConstraintMessage();

  /**
   * @brief compute the covariance of error in $T_BnBr$ given the covariance of errors in $T_WBr$ and $T_WBn$
   * $T_BnBr = T_WBn^{-1} T_WBr$
   * The error(perturbation) of  $T_WBr$ $T_WBn$ and $T_BnBr$ are defined by
   * okvis::Transformation::oplus and ominus.
   * @param T_WBr
   * @param cov_T_WBr
   * @param[out] cov_T_BnBr cov for error in $T_BnBr$.
   * @return
   */
  void computeRelativePoseCovariance(
      const okvis::kinematics::Transformation& T_WBr,
      const Eigen::Matrix<double, 6, 6>& cov_T_WBr,
      Eigen::Matrix<double, 6, 6>* cov_T_BnBr);

  NeighborConstraintInDatabase core_;

  // variables used for computing the weighting covariance for the constraint
  // in the case of odometry pose constraint. In the case of loop constraint,
  // the covariance is computed inside PnP solver.
  okvis::kinematics::Transformation T_WB_; // pose of this neighbor keyframe.
  // cov of T_WB
  Eigen::Matrix<double, 6, 6> cov_T_WB_;
  // cov(T_WBr, T_WB)
  Eigen::Matrix<double, 6, 6> cov_T_WBr_T_WB_;
};

/**
 * @brief The KeyframeInDatabase class is stored keyframe info in loop closure keyframe database.
 */
class KeyframeInDatabase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DELETE_COPY_CONSTRUCTORS(KeyframeInDatabase);

  KeyframeInDatabase();

  KeyframeInDatabase(size_t dbowId, uint64_t vioId, okvis::Time stamp,
                     const okvis::kinematics::Transformation& vio_T_WB,
                     const Eigen::Matrix<double, 6, 6>& cov_T_WB);

  void setOdometryConstraints(
      const std::vector<std::shared_ptr<NeighborConstraintMessage>>&
          odometryConstraintList) {
    constraintList_.reserve(odometryConstraintList.size());
    for (auto constraint : odometryConstraintList) {
      std::shared_ptr<okvis::NeighborConstraintInDatabase> dbConstraint(
          new okvis::NeighborConstraintInDatabase(constraint->core_));
      constraintList_.push_back(dbConstraint);
    }
  }

  const std::vector<std::shared_ptr<NeighborConstraintInDatabase>>&
  constraintList() const {
    return constraintList_;
  }

  void addLoopConstraint(
      std::shared_ptr<NeighborConstraintInDatabase>& loopConstraint) {
    loopConstraintList_.push_back(loopConstraint);
  }

  const cv::Mat frontendDescriptors() const {
     return frontendDescriptors_;
  }

  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
  landmarkPositionList() const {
    return landmarkPositionList_;
  }

  void setSquareRootInfo(size_t j,
                      const Eigen::Matrix<double, 6, 6>& squareRootInfo) {
    constraintList_.at(j)->squareRootInfo_ = squareRootInfo;
  }

  void setSquareRootInfoFromCovariance(size_t j,
                      const Eigen::Matrix<double, 6, 6>& covRawError);

  void setLandmarkPositionList(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          landmarkPositionList) {
    landmarkPositionList_ = landmarkPositionList;
  }

  void setFrontendDescriptors(cv::Mat frontendDescriptors) {
    frontendDescriptors_ = frontendDescriptors;
  }

 public:
  size_t dbowId_; ///< id used in DBoW vocabulary. Determined by the size of the KeyframeInDatabase list.
  uint64_t id_; ///< frontend keyframe id.
  okvis::Time stamp_;

  const okvis::kinematics::Transformation vio_T_WB_; ///< original vio estimated T_WB;
  const Eigen::Matrix<double, 6, 6> cov_vio_T_WB_;  ///< cov of $[\delta p, \delta \theta]$ provided by VIO.

 private:
  ///< If we do not construct the pose graph solver from scratches once in a
  /// while as in VINS Mono, then we do not need the constraint list.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> constraintList_; ///< odometry constraints.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> loopConstraintList_; ///< loop constraints.
  cv::Mat frontendDescriptors_; ///< landmark descriptors used in frontend. #columns is the descriptor size, #rows is for landmarks.
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      landmarkPositionList_;  ///< landmark positions expressed in the camera frame.
};

/**
 * @brief The LoopQueryKeyframeMessage class
 * Only one out of nframe will be used for querying keyframe database and
 * computing loop constraint. As a result, from the NCameraSystem, we only
 * need the camera intrinsic parameters, but not the extrinsic parameters.
 * We may reset the NCameraSystem for nframe_ when intrinsic parameters are
 * estimated online by the estimator. This should not disturb the frontend
 * feature matching which locks the estimator in matching features.
 */
class LoopQueryKeyframeMessage {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DELETE_COPY_CONSTRUCTORS(LoopQueryKeyframeMessage);

  LoopQueryKeyframeMessage();
  LoopQueryKeyframeMessage(uint64_t id, okvis::Time stamp,
                           const okvis::kinematics::Transformation& T_WB,
                           std::shared_ptr<const okvis::MultiFrame> multiframe);
  ~LoopQueryKeyframeMessage();

  std::shared_ptr<KeyframeInDatabase> toKeyframeInDatebase(size_t dbowId) const {
    return std::shared_ptr<KeyframeInDatabase>(
        new KeyframeInDatabase(dbowId, id_, stamp_, T_WB_, cov_T_WB_));
  }

  std::shared_ptr<const okvis::MultiFrame> NFrame() const {
    return nframe_;
  }

  const cv::Mat queryImage() const {
    return nframe_->image(kQueryCameraIndex);
  }

  std::shared_ptr<const cameras::CameraBase> cameraGeometry() const {
    return nframe_->geometry(kQueryCameraIndex);
  }

  const std::vector<std::shared_ptr<NeighborConstraintMessage>>&
  odometryConstraintList() const {
    return odometryConstraintList_;
  }

  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
  landmarkPositionList() const {
    return landmarkPositionList_;
  }

  ///< \brief gather descriptors for keypoints associated to landmarks.
  cv::Mat gatherFrontendDescriptors() const {
    return nframe_->copyDescriptorsAt(kQueryCameraIndex, keypointIndexForLandmarkList_);
  }

  ///< \brief get all descriptors for a view in nframe.
  cv::Mat getDescriptors() const {
    return nframe_->getDescriptors(kQueryCameraIndex);
  }

  std::vector<std::shared_ptr<NeighborConstraintMessage>>&
  odometryConstraintListMutable() {
    return odometryConstraintList_;
  }

  const std::vector<int>& keypointIndexForLandmarkList() const {
    return keypointIndexForLandmarkList_;
  }

  std::vector<int>& keypointIndexForLandmarkListMutable() {
    return keypointIndexForLandmarkList_;
  }

  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
  landmarkPositionListMutable() {
    return landmarkPositionList_;
  }

  void setLandmarkPositionList(
      const std::vector<Eigen::Vector4d,
                        Eigen::aligned_allocator<Eigen::Vector4d>>&
          landmarkPositionList) {
    landmarkPositionList_ = landmarkPositionList;
  }

  uint64_t id_;
  okvis::Time stamp_;
  okvis::kinematics::Transformation T_WB_;

  Eigen::Matrix<double, 6, 6> cov_T_WB_;  ///< cov of $[\delta p, \delta \theta]$
  const static size_t kQueryCameraIndex = 0u;

 private:
  /// @warn Do not hold on to nframe_ which has many images.
  std::shared_ptr<const okvis::MultiFrame> nframe_; ///< nframe contains the list of keypoints for each subframe, and the camera system info.

  std::vector<std::shared_ptr<NeighborConstraintMessage>> odometryConstraintList_;

  std::vector<int> keypointIndexForLandmarkList_; ///< Index of the keypoints with landmark positions.
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      landmarkPositionList_;  ///< landmark positions expressed in the body frame of this keyframe.

}; // LoopQueryKeyframeMessage
}  // namespace okvis
#endif  // INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
