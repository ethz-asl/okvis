/**
 * @file EstimatorLoopClosure.cpp
 * @brief implementation of estimator functions related to loop closure
 */

#include <map>
#include <memory>

#include <okvis/Estimator.hpp>
#include <okvis/KeyframeForLoopDetection.hpp>

namespace okvis {
bool Estimator::getLoopQueryKeyframeMessage(
    okvis::MultiFramePtr multiFrame,
    std::shared_ptr<okvis::LoopQueryKeyframeMessage> queryKeyframe) const {
  auto riter = statesMap_.rbegin();
  if (!riter->second.isKeyframe) {
    return false;
  }
  okvis::kinematics::Transformation T_WBr;
  get_T_WS(riter->first, T_WBr);
  queryKeyframe.reset(new okvis::LoopQueryKeyframeMessage(
                        riter->first, riter->second.timestamp,
                        T_WBr, multiFrame));

  for (++riter; riter != statesMap_.rend(); ++riter) {
    if (riter->second.isKeyframe) {
        okvis::kinematics::Transformation T_WBn;
        get_T_WS(riter->first, T_WBn);
        okvis::kinematics::Transformation T_BrBn = T_WBr.inverse() * T_WBn;
        std::shared_ptr<okvis::NeighborConstraintMessage> odometryConstraint(
              new okvis::NeighborConstraintMessage(riter->first, riter->second.timestamp, T_BrBn));
        odometryConstraint->core_.cov_T_BrB_.setIdentity();
        queryKeyframe->odometryConstraintList_.emplace_back(odometryConstraint);
    }
  }
  // add 3d landmarks anchored to query keyframe, corresponding 2d keypoint positions.
  // do not keep all keypoints which will waste much memory?

  return true;
}
} // namespace okvis
