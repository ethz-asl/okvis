#include "msckf/PointSharedData.hpp"
namespace msckf {
void PointSharedData::computePoseAndVelocityAtObservation() {
  for (auto& item : stateInfoForObservations_) {
    std::shared_ptr<const okvis::ceres::ParameterBlock> b = item.T_WBj_ptr;
    item.T_WBtij =
        std::static_pointer_cast<const okvis::ceres::PoseParameterBlock>(b)
            ->estimate();
  }
}

void PointSharedData::
    computePoseAndVelocitWithFirstEstimates() {
}

void PointSharedData::computeSharedJacobians() {}
}
