#ifndef MSCKF_TEST_SIMULATED_MOTION_FOR_PARALLAX_ANGLE_TEST_HPP_
#define MSCKF_TEST_SIMULATED_MOTION_FOR_PARALLAX_ANGLE_TEST_HPP_

#include <Eigen/Core>
#include <msckf/DirectionFromParallaxAngleJacobian.hpp>
#include <msckf/RemoveFromVector.hpp>
#include <msckf/VectorNormalizationJacobian.hpp>

namespace simul {
enum MotionType {
  Sideways = 0, // R_WC = I3
  Sideways_R_WC = 1, // R_WC != I3
  Forward,
  Backward,
  Mixed,
  AssociateObserver,
  Rotation,
  AntiRotation,
  ForwardWithCenterPoint,
};

std::ostream& operator<<(std::ostream& out, const MotionType value);

/**
 * @brief RotY If we rotate a coordinate frame A about its y axis for theta
 *    to obtain a coordinate frame B, then the coordinates of a point in B p_B
 *    and its coordinates in A are related by p_A = RotY * p_B.
 * @param theta
 * @return
 */
Eigen::Matrix3d RotY(double theta);

typedef std::vector<okvis::kinematics::Transformation,
Eigen::aligned_allocator<okvis::kinematics::Transformation>> TransformationList;

class SimulatedMotionForParallaxAngleTest {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  enum ProjectionListStatus {
    Healthy = 0,
    MainAnchorFailed = 1,
    AssociateAnchorFailed = 2,
  };

  SimulatedMotionForParallaxAngleTest(MotionType motion,
                                      bool addOutlierObservation);

  void dN_dp_WCmi(Eigen::Matrix3d* j) const;

  void dN_dtheta_WCmi(Eigen::Matrix3d* j) const;

  void dN_dp_WCai(Eigen::Matrix3d* j) const;

  void dN_dp_WCtij(Eigen::Matrix3d* j) const;

  void dN_dni(Eigen::Matrix<double, 3, 2>* j) const;

  void dN_dthetai(Eigen::Vector3d* j) const;


  const std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>&
  observations() const {
    return observationsxy1_;
  }

  const std::vector<okvis::kinematics::Transformation,
                    Eigen::aligned_allocator<okvis::kinematics::Transformation>>&
  T_WC_list() const {
    return T_WC_list_;
  }

  std::vector<size_t> anchorIndices() const {
    return anchorObservationIndices_;
  }

  const LWF::ParallaxAnglePoint& pap()
      const {
    return pap_;
  }

  msckf::DirectionFromParallaxAngleJacobian dfpaj_;
  ProjectionListStatus observationListStatus_;

 private:
  double computeCosParallaxAngle() const {
    return computeCosParallaxAngle(pW_, T_WCm_.r(), T_WCa_.r());
  }

  double computeCosParallaxAngle(const Eigen::Vector3d& pW,
                                 const Eigen::Vector3d& t_WCm,
                                 const Eigen::Vector3d& t_WCa) const;

  bool computeObservationXY1(const Eigen::Vector3d pW,
                             const okvis::kinematics::Transformation& T_WC,
                             Eigen::Vector3d* xy1, bool addNoise);

  void simulateSidewaysMotion(double a, bool addOutlier);

  okvis::kinematics::Transformation T_WCm_;
  okvis::kinematics::Transformation T_WCa_;
  okvis::kinematics::Transformation T_WCj_;
  Eigen::Vector3d pW_;
  double cosParallaxAngle_;
  LWF::ParallaxAnglePoint pap_;
  Eigen::Vector3d Nij_;
  const MotionType motionType_;

  std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
      observationsxy1_;
  TransformationList T_WC_list_;
  std::vector<size_t> anchorObservationIndices_;
};
}
#endif // MSCKF_TEST_SIMULATED_MOTION_FOR_PARALLAX_ANGLE_TEST_HPP_
