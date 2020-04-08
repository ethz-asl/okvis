#ifndef INCLUDE_OKVIS_LOOP_CLOSURE_METHOD_HPP_
#define INCLUDE_OKVIS_LOOP_CLOSURE_METHOD_HPP_

#include <memory>

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

  explicit LoopClosureMethod(const LoopClosureParameters& parameters);

  virtual ~LoopClosureMethod();

  /**
   * @brief detectLoop detect loop and add keyframe to vocabulary and keyframe database.
   * @param[in] queryKeyframe
   * @param[out] queryKeyframeInDB
   * @param[out] loopFrameAndMatches
   * @return true if loop frame(s) detected.
   */
  virtual bool detectLoop(
      std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe,
      std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB,
      std::shared_ptr<LoopFrameAndMatches> loopFrameAndMatches);

  /**
   * @brief addConstraintsAndOptimize add constraints to the pose graph,
   * remove outliers, and optimize
   * @param[in] queryKeyframeInDB
   * @param[in] loopKeyframe
   * @return true if optimization is performed.
   */
  virtual bool addConstraintsAndOptimize(
      std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB,
      std::shared_ptr<LoopFrameAndMatches> loopKeyframe);

  const static size_t kMethodId = 0u;
 private:
  LoopClosureParameters loopClosureParameters_;
  std::vector<std::shared_ptr<okvis::KeyframeInDatabase>> db_frames_;
};
}  // namespace okvis

#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_METHOD_HPP_
