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

  LoopFrameAndMatches(uint64_t id, uint64_t queryKeyframeId,
                      const okvis::kinematics::Transformation& T_BlBq);

  ~LoopFrameAndMatches();

  uint64_t id_;
  uint64_t queryKeyframeId_;
  okvis::kinematics::Transformation
      T_BlBq_;  ///< Bl loop frame, Bq query keyframe.
  okvis::kinematics::Transformation
      T_WB_;  ///< Pose of the found loop frame in the world frame of the latest
             ///< pose graph.
  Eigen::Matrix<double, 6, 6>
      cov_T_BlBq_;  ///< cov of T_BlBq computed inside PnP
  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      matchedKeypoints_;
  std::vector<int> matchedKeypointIndexInQueryFrame_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      matchedLandmarks_;  // positions are expressed in the loop frame.
};
}  // namespace okvis
#endif  // INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
