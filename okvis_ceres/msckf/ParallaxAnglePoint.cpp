#include <msckf/ParallaxAnglePoint.hpp>
#include <msckf/ceres/tiny_solver.h>
#include <msckf/BearingResiduals.hpp>
#include <msckf/PointLandmarkModels.hpp>


namespace LWF {
bool ParallaxAnglePoint::initializePosition(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d> >
        &observationsxy1,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation> >
        &T_WC_list,
    const std::vector<int> &anchorIndices) {
  Eigen::Vector3d d_m = observationsxy1[anchorIndices[0]].normalized();
  Eigen::Vector3d d_a = observationsxy1[anchorIndices[1]].normalized();
  Eigen::Vector3d W_d_m = T_WC_list[anchorIndices[0]].C() * d_m;
  Eigen::Vector3d W_d_a = T_WC_list[anchorIndices[1]].C() * d_a;
  double cos_theta = W_d_m.dot(W_d_a);
  n_.setFromVector(d_m);
  theta_.setFromCosine(cos_theta);
  return true;
}

bool ParallaxAnglePoint::optimizePosition(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>&
        observationsxy1,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WC_list,
    const std::vector<int>& anchorIndices) {
  std::shared_ptr<msckf::SimplePointSharedData> pointDataPtr(
        new msckf::SimplePointSharedData());
  pointDataPtr->T_WC_list = T_WC_list;
  pointDataPtr->anchorIndices = anchorIndices;
  pointDataPtr->unitBearingList.reserve(observationsxy1.size());
  for (auto& xy1 : observationsxy1) {
    pointDataPtr->unitBearingList.emplace_back(xy1.normalized());
  }
  msckf::BearingResiduals f(pointDataPtr, 0.01);
  Eigen::Matrix<double, 6, 1> x;
  copy(&x);
  msckf::ceres::TinySolver<msckf::BearingResiduals> solver;
  solver.options.max_num_iterations = 15;
  msckf::ParallaxAngleParameterization localPap;
  solver.localParameterization_ = &localPap;
  solver.Solve(f, &x);
  set(x.data());
  return true;
}
} // namespace LWF
