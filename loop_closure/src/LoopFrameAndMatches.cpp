#include <okvis/LoopFrameAndMatches.hpp>
namespace okvis {
LoopFrameAndMatches::LoopFrameAndMatches() {}

LoopFrameAndMatches::LoopFrameAndMatches(
    uint64_t id, okvis::Time stamp, size_t dbowId, uint64_t queryKeyframeId,
    okvis::Time queryKeyframeStamp, size_t queryKeyframeDbowId,
    const okvis::kinematics::Transformation& T_BlBq)
    : id_(id),
      stamp_(stamp),
      dbowId_(dbowId),
      queryKeyframeId_(queryKeyframeId),
      queryKeyframeStamp_(queryKeyframeStamp),
      queryKeyframeDbowId_(queryKeyframeDbowId),
      T_BlBq_(T_BlBq) {}

LoopFrameAndMatches::~LoopFrameAndMatches() {}
}  // namespace okvis
