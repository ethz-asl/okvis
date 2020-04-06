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

  virtual std::shared_ptr<LoopFrameAndMatches> detectLoop(
      std::shared_ptr<KeyframeForLoopDetection> queryKeyframe);

  virtual bool addConstraintsAndOptimize(
      std::shared_ptr<KeyframeForLoopDetection> queryKeyframe,
      std::shared_ptr<LoopFrameAndMatches> loopKeyframe);
  const static size_t kMethodId = 0u;
 private:
  LoopClosureParameters loopClosureParameters_;
};
}  // namespace okvis

#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_METHOD_HPP_
