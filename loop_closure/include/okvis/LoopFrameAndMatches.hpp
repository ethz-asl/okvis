#ifndef INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
#define INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
class LoopFrameAndMatches {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LoopFrameAndMatches();
  ~LoopFrameAndMatches();

  uint64_t id;
  uint64_t queryKeyframeId;
  okvis::kinematics::Transformation
      T_WB;  ///< Pose of the found loop frame in the world frame of the latest
             ///< pose graph.
  okvis::kinematics::Transformation
      T_BlBq;  ///< Bl loop frame, Bq query keyframe
  Eigen::Matrix<double, 6, 6>
      cov_T_BlBq;  ///< cov of T_BlBq computed inside PnP
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      matchedKeypoints;
  std::vector<int> matchedKeypointIndexInQueryFrame;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      matchedLandmarks;  // positions are expressed in the loop frame.
};
}  // namespace okvis
#endif  // INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
