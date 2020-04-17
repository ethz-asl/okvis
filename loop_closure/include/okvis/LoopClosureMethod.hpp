#ifndef INCLUDE_OKVIS_LOOP_CLOSURE_METHOD_HPP_
#define INCLUDE_OKVIS_LOOP_CLOSURE_METHOD_HPP_

#include <memory>
#include <unordered_map>

#include <okvis/KeyframeForLoopDetection.hpp>
#include <okvis/LoopClosureParameters.hpp>
#include <okvis/LoopFrameAndMatches.hpp>

namespace okvis {
/**
 * @brief The LoopClosureMethod class implements loop closure detection and pose
 * graph optimization. It suits to be subclassed.
 */
class LoopClosureMethod {
 public:
  LoopClosureMethod();

  explicit LoopClosureMethod(std::shared_ptr<const okvis::LoopClosureParameters> parameters);

  virtual ~LoopClosureMethod();

  /**
   * @brief detectLoop detect loop and add keyframe to vocabulary and keyframe database.
   * Also convert the odometry constraints in queryKeyframe to constraint list in queryKeyframeInDB,
   * The constraint list of queryKeyframeInDB also includes the newly found loop constraint.
   * @param[in] queryKeyframe
   * @param[out] queryKeyframeInDB
   * @param[out] loopFrameAndMatches
   * @return true if loop frame(s) detected.
   */
  virtual bool detectLoop(
      std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe,
      std::shared_ptr<KeyframeInDatabase>& queryKeyframeInDB,
      std::shared_ptr<LoopFrameAndMatches>& loopFrameAndMatches);

  /**
   * @brief addConstraintsAndOptimize add constraints to the pose graph,
   * remove outliers, and optimize
   * @warning This module needs to handle the situation of dropped odometry
   * messages. One implication is that transmitting only relative pose for
   * odometry constraints does not work.
   * @param[in] queryKeyframeInDB
   * @param[in] loopKeyframe
   * @return true if optimization is performed.
   */
  virtual bool addConstraintsAndOptimize(
      std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB,
      std::shared_ptr<LoopFrameAndMatches> loopKeyframe);

  /**
   * @brief initialize a keyframe in database from a query keyframe.
   * The main job is to fill the covariance for neighboring odometry constraints.
   * This function is virtual because the covariance depends on the PGO module.
   * @param queryKeyframe
   * @return queryKeyframeInDB
   */
  virtual std::shared_ptr<KeyframeInDatabase> initializeKeyframeInDatabase(
      size_t dbowId,
      std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe) const;

  inline std::vector<std::shared_ptr<okvis::KeyframeInDatabase>>
  getFrameDatabasePtr() const {
    return db_frames_;
  }

 protected:
  std::vector<std::shared_ptr<okvis::KeyframeInDatabase>> db_frames_;
  std::unordered_map<uint64_t, size_t> vioIdToDbowId_;
};
}  // namespace okvis

#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_METHOD_HPP_
