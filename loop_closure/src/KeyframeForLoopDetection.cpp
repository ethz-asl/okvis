#include <okvis/KeyframeForLoopDetection.hpp>

namespace okvis {
NeighborConstraintInDatabase::NeighborConstraintInDatabase() {}

NeighborConstraintInDatabase::NeighborConstraintInDatabase(
    uint64_t id, okvis::Time stamp,
    const okvis::kinematics::Transformation& T_BrB,
    PoseConstraintType type) :
  id_(id), stamp_(stamp), T_BrB_(T_BrB), type_(type) {

}

NeighborConstraintInDatabase::~NeighborConstraintInDatabase() {}

NeighborConstraintMessage::NeighborConstraintMessage() {}

NeighborConstraintMessage::NeighborConstraintMessage(
    uint64_t id, okvis::Time stamp,
    const okvis::kinematics::Transformation& T_BrB,
    PoseConstraintType type) :
  core_(id, stamp, T_BrB, type) {

}

NeighborConstraintMessage::~NeighborConstraintMessage() {}

LoopQueryKeyframeMessage::LoopQueryKeyframeMessage() {}

LoopQueryKeyframeMessage::LoopQueryKeyframeMessage(uint64_t id, okvis::Time stamp,
                         const okvis::kinematics::Transformation& T_WB,
                         okvis::MultiFramePtr multiframe)
    : id_(id), stamp_(stamp), T_WB_(T_WB), nframe_(multiframe) {}

LoopQueryKeyframeMessage::~LoopQueryKeyframeMessage() {}
}  // namespace okvis
