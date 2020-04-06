#ifndef INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#define INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <okvis/MultiFrame.hpp>

namespace okvis {
class NeighborPoseConstraint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  NeighborPoseConstraint();
  ~NeighborPoseConstraint();

  uint64_t id;
  okvis::Time stamp;
  // Br is a body frame for reference, B body frame of this neighbor.
  okvis::kinematics::Transformation T_BrB;

  // variables used for computing the weighting covariance for the constraint
  // in the case of odometry pose constraint. In the case of loop constraint,
  // the covariance is computed inside PnP solver.
  // cov of T_WB
  Eigen::Matrix<double, 6, 6> cov_T_WB;
  // cov(T_WBr, T_WB)
  Eigen::Matrix<double, 6, 6> cov_T_WBr_T_WB;

  // cov of T_BrB is used for weighting the pose constraint.
  Eigen::Matrix<double, 6, 6> cov_T_BrB;
};

class KeyframeForLoopDetection {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  KeyframeForLoopDetection();
  ~KeyframeForLoopDetection();

  uint64_t id;
  okvis::Time stamp;
  okvis::kinematics::Transformation T_WB;
  Eigen::Matrix<double, 6, 6> cov_T_WB;  // cov of $[\delta p, \delta \theta]$
  std::shared_ptr<okvis::MultiFrame> nframe;
  std::vector<std::shared_ptr<NeighborPoseConstraint>> odometryConstraintList;
  std::vector<int> keypointIndexForLandmarkList;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      landmarkPositionList;
};

}  // namespace okvis
#endif  // INCLUDE_OKVIS_KEYFRAME_FOR_LOOP_DETECTION_HPP_
