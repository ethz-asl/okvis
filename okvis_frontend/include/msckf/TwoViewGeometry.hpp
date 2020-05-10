#ifndef INCLUDE_MSCKF_TWO_VIEW_GEOMETRY_HPP_
#define INCLUDE_MSCKF_TWO_VIEW_GEOMETRY_HPP_
#include <okvis/kinematics/Transformation.hpp>
#include <Eigen/Core>

namespace okvis {
class TwoViewGeometry {
 public:
  static double computeErrorEssentialMat(okvis::kinematics::Transformation T_ji,
                                         Eigen::Vector3d bearing_i,
                                         Eigen::Vector3d bearing_j, double fi,
                                         double fj, double sigmai = 1.0,
                                         double sigmaj = 1.0);

  /**
   * @brief deviationFromEpipolarLine compute deviation in pixels from the
   * epipolar line. see
   * https://github.com/opencv/opencv/blob/master/modules/calib3d/src/fundam.cpp
   * @param E_ji t_ji X R_ji. Epipolar constraint is p_j' * t_ji X R_ji * p_i = 0.
   * @param bearing_i [x, y, 1] undistorted image coordinate at z=1 for point in
   * image i
   * @param bearing_j [x, y, 1] undistorted image coordinate at z=1 for point in
   * image j
   * @param focal_length nominal focal length to convert the epipolar line error
   * into error of pixel unit.
   * @return squared distance to epipolar line. Distance has a unit of pixels.
   */
  static double computeErrorEssentialMat(Eigen::Matrix3d E_ji,
                                         Eigen::Vector3d bearing_i,
                                         Eigen::Vector3d bearing_j, double fi,
                                         double fj, double sigmai = 1.0,
                                         double sigmaj = 1.0);
};
}  // namespace okvis

#endif  // INCLUDE_MSCKF_TWO_VIEW_GEOMETRY_HPP_
