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

  const Eigen::Matrix<double, 6, 6>& relativePoseCovariance() const {
    return cov_T_BlBq_;
  }

  const Eigen::Matrix<double, 6, 6>& poseCovariance() const {
    return cov_vio_T_WB_;
  }

  void setRelativePoseCovariance(const Eigen::Matrix<double, 6, 6>& cov_T_BlBq) {
     cov_T_BlBq_ = cov_T_BlBq;
  }

  void setPoseCovariance(const Eigen::Matrix<double, 6, 6>& cov_T_WB) {
    cov_vio_T_WB_ = cov_T_WB;
  }

  uint64_t id_;
  uint64_t queryKeyframeId_;
  okvis::kinematics::Transformation
      T_BlBq_;  ///< Bl loop frame, Bq query keyframe.
  okvis::kinematics::Transformation
      T_WB_;  ///< Pose of the found loop frame in the world frame of the latest
             ///< pose graph.
 private:
  // We do not compute cov of the pose provided by PGO because
  // it is expensive for optimization-based PGO.
  Eigen::Matrix<double, 6, 6> cov_vio_T_WB_; ///< cov of the pose of the loop frame provided by VIO.

  Eigen::Matrix<double, 6, 6>
      cov_T_BlBq_;  ///< cov of T_BlBq computed inside PnP, l loop kf, q query kf.

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      matchedKeypoints_;
  std::vector<int> matchedKeypointIndexInQueryFrame_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      matchedLandmarks_;  // positions are expressed in the loop frame.
};
}  // namespace okvis
#endif  // INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
