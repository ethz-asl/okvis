#include <okvis/LoopClosureMethod.hpp>
namespace okvis {
LoopClosureMethod::LoopClosureMethod() {}

LoopClosureMethod::LoopClosureMethod(const LoopClosureParameters& parameters)
    : loopClosureParameters_(parameters) {}

LoopClosureMethod::~LoopClosureMethod() {}

std::shared_ptr<LoopFrameAndMatches> LoopClosureMethod::detectLoop(
    std::shared_ptr<KeyframeForLoopDetection> /*queryKeyframe*/) {
  std::shared_ptr<LoopFrameAndMatches> loopFrameAndMatches;
  return loopFrameAndMatches;
}

bool LoopClosureMethod::addConstraintsAndOptimize(
    std::shared_ptr<KeyframeForLoopDetection> /*queryKeyframe*/,
    std::shared_ptr<LoopFrameAndMatches> /*loopKeyframe*/) {
  return true;
}

}  // namespace okvis
