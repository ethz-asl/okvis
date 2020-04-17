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
 NeighborConstraintInDatabase(
     uint64_t id, okvis::Time stamp,
     const okvis::kinematics::Transformation& T_BrB,
     PoseConstraintType type);
 ~NeighborConstraintInDatabase();

 uint64_t id_;
 okvis::Time stamp_;

 // Br is a body frame for reference, B body frame of this neighbor.
 okvis::kinematics::Transformation T_BrB_;

 PoseConstraintType type_;

 // covariance of the between factor unwhitened/raw error due to measurement noise.
 // It depends on definitions of the between factor and errors of the measurement.
 // e.g., gtsam::BetweenFactor<Pose3>==log(T_z^{-1}T_x^{-1}T_y) and
 // error of T_z is defined by T_z = Pose3::Retraction(\hat{T}_z, \delta).
 Eigen::Matrix<double, 6, 6> covRawError_;
};

class NeighborConstraintMessage {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeighborConstraintMessage();
  NeighborConstraintMessage(
      uint64_t id, okvis::Time stamp,
      const okvis::kinematics::Transformation& T_BrB,
      const okvis::kinematics::Transformation& T_WB,
      PoseConstraintType type = PoseConstraintType::Odometry);
  ~NeighborConstraintMessage();

  /**
   * @brief compute the covariance of error in $T_BrBn$ given the covariance of errors in $T_WBr$ and $T_WBn$
   * $T_BrBn = T_WBr^{-1} T_WBn$
   * The error(perturbation) of  $T_WBr$ $T_WBn$ and $T_BrBn$ are defined by
   * okvis::Transformation::oplus and ominus.
   * The computed covariance will be put in core_.cov_T_BrB_.
   * @param T_WBr
   * @param cov_T_WBr
   * @param cov_T_BrB cov for error in $T_BrBn$.
   * @return
   */
  void computeRelativePoseCovariance(
      const okvis::kinematics::Transformation& T_WBr,
      const Eigen::Matrix<double, 6, 6>& cov_T_WBr,
      Eigen::Matrix<double, 6, 6>* cov_T_BrB);

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
                     const okvis::kinematics::Transformation& T_BC,
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

  const cv::Mat frontendDescriptors() const {
     return frontendDescriptors_;
  }

  void setCovRawError(size_t j,
                      const Eigen::Matrix<double, 6, 6>& covRawBetweenError) {
    constraintList_.at(j)->covRawError_ = covRawBetweenError;
  }

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
  okvis::kinematics::Transformation T_WB_; ///< T_WB estimated by pose graph optimizer.

  const okvis::kinematics::Transformation vio_T_WB_; ///< original vio estimated T_WB;
  const okvis::kinematics::Transformation T_BC_; ///< T_BC body frame to left camera.
  const Eigen::Matrix<double, 6, 6> cov_vio_T_WB_;  ///< cov of $[\delta p, \delta \theta]$ provided by VIO.

 private:
  ///< If we do not construct the pose graph solver from scratches once in a
  /// while as in VINS Mono, then we do not need the constraint list.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> constraintList_; ///< odometry or loop constraints.

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

  std::shared_ptr<KeyframeInDatabase> toKeyframeInDatebase(size_t dbowId) {
    return std::shared_ptr<KeyframeInDatabase>(
        new KeyframeInDatabase(dbowId, id_, stamp_, T_WB_, T_BC_, cov_T_WB_));
  }

  std::shared_ptr<const okvis::MultiFrame> NFrame() const {
    return nframe_;
  }

  const cv::Mat queryImage() const {
    return nframe_->image(kQueryCameraIndex_);
  }

  std::shared_ptr<const cameras::CameraBase> cameraGeometry() const {
    return nframe_->geometry(kQueryCameraIndex_);
  }

  const std::vector<std::shared_ptr<NeighborConstraintMessage>>&
  odometryConstraintList() const {
    return odometryConstraintList_;
  }

  const std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>&
  landmarkPositionList() const {
    return landmarkPositionList_;
  }

  cv::Mat gatherFrontendDescriptors() const {
    return nframe_->copyDescriptorsAt(kQueryCameraIndex_, keypointIndexForLandmarkList_);
  }

  std::vector<std::shared_ptr<NeighborConstraintMessage>>&
  odometryConstraintListMutable() {
    return odometryConstraintList_;
  };

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
  okvis::kinematics::Transformation T_BC_; ///< latest estimate by VIO for transform between camera and body frame.

  Eigen::Matrix<double, 6, 6> cov_T_WB_;  ///< cov of $[\delta p, \delta \theta]$

 private:
  /// @warn Do not hold on to nframe_ which has many images.
  std::shared_ptr<const okvis::MultiFrame> nframe_; ///< nframe contains the list of keypoints for each subframe, and the camera system info.

  std::vector<std::shared_ptr<NeighborConstraintMessage>> odometryConstraintList_;

  std::vector<int> keypointIndexForLandmarkList_; ///< Index of the keypoints with landmark positions.
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      landmarkPositionList_;  ///< landmark positions expressed in the camera frame
  const size_t kQueryCameraIndex_ = 0u;
}; // LoopQueryKeyframeMessage
}  // namespace okvis
#endif  // INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
