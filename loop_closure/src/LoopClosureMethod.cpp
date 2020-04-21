#include <okvis/LoopClosureMethod.hpp>
namespace okvis {
LoopClosureMethod::LoopClosureMethod() {}

LoopClosureMethod::~LoopClosureMethod() {}

bool LoopClosureMethod::detectLoop(
    std::shared_ptr<const LoopQueryKeyframeMessage> queryKeyframe,
    std::shared_ptr<KeyframeInDatabase>& /*queryKeyframeInDB*/,
    std::shared_ptr<LoopFrameAndMatches>& /*loopFrameAndMatches*/) {
  queryKeyframe.reset();
  return false;
}

bool LoopClosureMethod::addConstraintsAndOptimize(
    const KeyframeInDatabase& /*queryKeyframe*/,
    std::shared_ptr<const LoopFrameAndMatches> loopKeyframe,
    PgoResult& /*pgoResult*/) {
  if (loopKeyframe) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<KeyframeInDatabase>
LoopClosureMethod::initializeKeyframeInDatabase(
    size_t dbowId, const LoopQueryKeyframeMessage& queryKeyframe) const {
  return queryKeyframe.toKeyframeInDatebase(dbowId);
}

}  // namespace okvis
