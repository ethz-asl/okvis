#include <okvis/LoopFrameAndMatches.hpp>
namespace okvis {
LoopFrameAndMatches::LoopFrameAndMatches() {}

LoopFrameAndMatches::LoopFrameAndMatches(
    uint64_t id, uint64_t queryKeyframeId,
    const okvis::kinematics::Transformation& T_BlBq)
    : id_(id),
      queryKeyframeId_(queryKeyframeId),
      T_BlBq_(T_BlBq) {}

LoopFrameAndMatches::~LoopFrameAndMatches() {}
}  // namespace okvis
