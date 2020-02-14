#include <msckf/ParallaxAnglePoint.hpp>
#include <ceres/tiny_solver.h>


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

  bool optimizedOk = optimizePosition(
        observationsxy1, T_WC_list, anchorIndices);
  return optimizedOk;
}

bool ParallaxAnglePoint::optimizePosition(
    const std::vector<Eigen::Vector3d,
                      Eigen::aligned_allocator<Eigen::Vector3d>>&
        observationsxy1,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WC_list,
    const std::vector<int>& anchorIndices) {
  return true;
}
} // namespace LWF
