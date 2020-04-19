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
    std::shared_ptr<const LoopFrameAndMatches> loopKeyframe) {
  if (loopKeyframe) {
    return true;
  } else {
    return false;
  }
}

std::shared_ptr<KeyframeInDatabase> LoopClosureMethod::initializeKeyframeInDatabase(
    size_t dbowId,
    const LoopQueryKeyframeMessage& queryKeyframe) const {
  std::shared_ptr<KeyframeInDatabase> queryKeyframeInDB = queryKeyframe.toKeyframeInDatebase(dbowId);
  queryKeyframeInDB->setOdometryConstraints(queryKeyframe.odometryConstraintList());
  queryKeyframeInDB->setLandmarkPositionList(queryKeyframe.landmarkPositionList());
  queryKeyframeInDB->setFrontendDescriptors(queryKeyframe.gatherFrontendDescriptors());
  return queryKeyframeInDB;
}

}  // namespace okvis
