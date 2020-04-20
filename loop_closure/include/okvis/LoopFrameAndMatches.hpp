#ifndef INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
#define INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
#include <vector>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <okvis/kinematics/Transformation.hpp>
#include <okvis/Time.hpp>

namespace okvis {
class LoopFrameAndMatches {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  LoopFrameAndMatches();

  LoopFrameAndMatches(uint64_t id, okvis::Time stamp, size_t dbowId,
                      uint64_t queryKeyframeId, okvis::Time queryKeyframeStamp,
                      size_t queryKeyframeDbowId,
                      const okvis::kinematics::Transformation& T_BlBq);

  ~LoopFrameAndMatches();

  const Eigen::Matrix<double, 6, 6>& relativePoseCovariance() const {
    return cov_T_BlBq_;
  }

  const Eigen::Matrix<double, 6, 6>& relativePoseSqrtInfo() const {
    return squareRootInfo_T_BlBq_;
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

  void setRelativePoseSqrtInfo(const Eigen::Matrix<double, 6, 6>& sqrtInfo_Bl_Pose_Bq) {
    squareRootInfo_T_BlBq_ = sqrtInfo_Bl_Pose_Bq;
  }

  uint64_t id_; ///< id of the loop keyframe earlier assigned by vio.
  okvis::Time stamp_;
  size_t dbowId_; ///< id of the loop keyframe in keyframe dbow database.

  uint64_t queryKeyframeId_; ///< id of the query keyframe assigned by vio.
  okvis::Time queryKeyframeStamp_;
  size_t queryKeyframeDbowId_; ///< id of the query keyframe in keyframe dbow database.

  okvis::kinematics::Transformation
      T_BlBq_;  ///< Bl loop frame, Bq query keyframe.

  okvis::kinematics::Transformation
      pgo_T_WBl_;  ///< Pose of the found loop frame in the world frame of the latest
             ///< pose graph.
 private:
  // We do not compute cov of the pose provided by PGO because
  // it is expensive for optimization-based PGO.
  Eigen::Matrix<double, 6, 6> cov_vio_T_WB_; ///< cov of the pose of the loop frame provided by VIO.

  Eigen::Matrix<double, 6, 6>
      cov_T_BlBq_;  ///< cov of T_BlBq computed inside PnP, l loop kf, q query kf.

  Eigen::Matrix<double, 6, 6>
      squareRootInfo_T_BlBq_;  ///< square root L' of the info matrix $\Lambda$
                               ///< for the error in T_BlBq such that $LL'=
                               ///< \Lambda$.

  std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>
      matchedKeypoints_;
  std::vector<int> matchedKeypointIndexInQueryFrame_;
  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      matchedLandmarks_;  // positions are expressed in the loop frame.
};
}  // namespace okvis
#endif  // INCLUDE_OKVIS_LOOP_FRAME_AND_MATCHES_HPP_
