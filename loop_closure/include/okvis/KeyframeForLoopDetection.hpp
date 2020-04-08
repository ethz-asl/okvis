#ifndef INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#define INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <okvis/MultiFrame.hpp>
#include <okvis/class_marcos.hpp>

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

 // cov of T_BrB is used for weighting the pose constraint.
 Eigen::Matrix<double, 6, 6> cov_T_BrB_;
};

class NeighborConstraintMessage {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeighborConstraintMessage();
  NeighborConstraintMessage(
      uint64_t id, okvis::Time stamp,
      const okvis::kinematics::Transformation& T_BrB,
      PoseConstraintType type = PoseConstraintType::Odometry);
  ~NeighborConstraintMessage();

  NeighborConstraintInDatabase core_;

  // variables used for computing the weighting covariance for the constraint
  // in the case of odometry pose constraint. In the case of loop constraint,
  // the covariance is computed inside PnP solver.
  // cov of T_WB
  Eigen::Matrix<double, 6, 6> cov_T_WB_;
  // cov(T_WBr, T_WB)
  Eigen::Matrix<double, 6, 6> cov_T_WBr_T_WB_;
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
                           okvis::MultiFramePtr multiframe);
  ~LoopQueryKeyframeMessage();

  uint64_t id_;
  okvis::Time stamp_;
  okvis::kinematics::Transformation T_WB_;
  okvis::kinematics::Transformation T_BC_; ///< latest estimate for transform between camera and body frame.
  /// @warn Do not hold on to nframe_ which has many images.
  std::shared_ptr<okvis::MultiFrame> nframe_; ///< nframe contains the list of keypoints for each subframe, and the camera system info.

  Eigen::Matrix<double, 6, 6> cov_T_WB_;  ///< cov of $[\delta p, \delta \theta]$

  std::vector<std::shared_ptr<NeighborConstraintMessage>> odometryConstraintList_;

  std::vector<int> keypointIndexForLandmarkList_; ///< Index of the keypoints with landmark positions.
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      landmarkPositionList_;  ///< landmark positions expressed in the camera frame
};

/**
 * @brief The KeyframeInDatabase class is stored keyframe info in loop closure keyframe database.
 */
class KeyframeInDatabase {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DELETE_COPY_CONSTRUCTORS(KeyframeInDatabase);

  size_t dbowId_; ///< id used in DBoW vocabulary. Determined by the size of the KeyframeInDatabase list.
  uint64_t id_; ///< frontend keyframe id.
  okvis::Time stamp_;
  okvis::kinematics::Transformation T_WB_;
  Eigen::Matrix<double, 6, 6> cov_T_WB_;  ///< cov of $[\delta p, \delta \theta]$

  ///< If we do not construct the pose graph solver from scratches once in a
  /// while as in VINS Mono, then we do not need the constraint list.
  std::vector<std::shared_ptr<NeighborConstraintInDatabase>> constraintList_; ///< odometry or loop constraints.

  cv::Mat frontendDescriptors_; ///< landmark descriptors used in frontend. #columns is the descriptor size, #rows is for landmarks.
  std::vector<Eigen::Vector4d, Eigen::aligned_allocator<Eigen::Vector4d>>
      landmarkPositionList_;  ///< landmark positions expressed in the camera frame.
};

}  // namespace okvis
#endif  // INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
