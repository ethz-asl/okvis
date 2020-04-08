#include <okvis/LoopClosureMethod.hpp>
namespace okvis {
LoopClosureMethod::LoopClosureMethod() {}

LoopClosureMethod::LoopClosureMethod(const LoopClosureParameters& parameters)
    : loopClosureParameters_(parameters) {}

LoopClosureMethod::~LoopClosureMethod() {}

bool LoopClosureMethod::detectLoop(
    std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe,
    std::shared_ptr<KeyframeInDatabase> /*queryKeyframeInDB*/,
    std::shared_ptr<LoopFrameAndMatches> /*loopFrameAndMatches*/) {
  queryKeyframe->nframe_.reset();
  return false;
}

bool LoopClosureMethod::addConstraintsAndOptimize(
    std::shared_ptr<KeyframeInDatabase> /*queryKeyframe*/,
    std::shared_ptr<LoopFrameAndMatches> loopKeyframe) {
  if (loopKeyframe) {
    return true;
  } else {
    return false;
  }
}

}  // namespace okvis
