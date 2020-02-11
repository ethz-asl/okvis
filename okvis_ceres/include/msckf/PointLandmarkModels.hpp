#ifndef INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_
#define INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <ceres/ceres.h>
#include <msckf/ParallaxAnglePoint.hpp>
namespace msckf {
// [x, y, z, w] usually expressed in the world frame.
class HomogeneousPointParameterization
{
public:
 static const int kModelId = 0;
 static const int kGlobalDim = 4;
 static const int kLocalDim = 3;
 template <class Scalar>
 static Eigen::Matrix<Scalar, 4, 1> bearingVectorInWorld(
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>&
         pairT_WBj,
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>*
         /*pairT_WBm*/,
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>*
         /*pairT_WBa*/,
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>&
         pairT_BC,
     const Scalar* parameters) {
   Eigen::Map<const Eigen::Matrix<Scalar, 4, 1>> hp_W(parameters);
   const Eigen::Quaternion<Scalar>& q_WB = pairT_WBj.first;
   const Eigen::Matrix<Scalar, 3, 1>& t_WB_W = pairT_WBj.second;

   const Eigen::Quaternion<Scalar>& q_BC = pairT_BC.first;
   const Eigen::Matrix<Scalar, 3, 1>& t_BC_B = pairT_BC.second;

   Eigen::Matrix<Scalar, 3, 3> C_WB = q_WB.toRotationMatrix();
   Eigen::Matrix<Scalar, 3, 1> t_WC_W = C_WB * t_BC_B + t_WB_W;
   Eigen::Matrix<Scalar, 4, 1> hp_CP_W = hp_W;
   hp_CP_W.template head<3>() -= hp_W[3] * t_WC_W;
   return hp_CP_W;
 }

 template <class Scalar>
 static Eigen::Matrix<Scalar, 4, 1> bearingVectorInCamera(
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>&
         pairT_WBj,
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>*
         pairT_WBm,
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>*
         pairT_WBa,
     const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>&
         pairT_BC,
     const Scalar* parameters) {
   Eigen::Matrix<Scalar, 4, 1> hp_CP_W = bearingVectorInWorld(
       pairT_WBj, pairT_WBm, pairT_WBa, pairT_BC, parameters);
   const Eigen::Quaternion<Scalar>& q_BC = pairT_BC.first;
   Eigen::Matrix<Scalar, 3, 3> C_BC = q_BC.toRotationMatrix();
   const Eigen::Quaternion<Scalar>& q_WB = pairT_WBj.first;
   Eigen::Matrix<Scalar, 3, 3> C_WB = q_WB.toRotationMatrix();
   Eigen::Matrix<Scalar, 4, 1> hp_CP_C = hp_CP_W;
   hp_CP_C.template head<3>() = (C_WB * C_BC).transpose() * hp_CP_W.template head<3>();
   return hp_CP_C;
 }
};

// Expressed in an anchor camera frame [\alpha, \beta, 1, \rho] = [x, y, z, w]/z.
class InverseDepthParameterization
{
public:
  static const int kModelId = 1;
  static const int kGlobalDim = 4;
  static const int kLocalDim = 3;

  template <class Scalar>
  static Eigen::Matrix<Scalar, 4, 1> bearingVectorInWorld(
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_WBj,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBm,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBa,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_BC,
          const Scalar* parameters) {

  }

};

// [x, y, z, w, c, s]
// [x, y, z, w] is the quaternion underlying unit bearing vector n such that
// n = q(w, x, y, z) * [0, 0, 1]'.
// c, s are cos(theta) and sin(theta) where \theta is the parallax angle.
class ParallaxAngleParameterization final: public ::ceres::LocalParameterization {
public:
  static const int kModelId = 2;
  static const int kGlobalDim = 6;
  static const int kLocalDim = 3;

  // compute dNij_d(n_i, theta_i)
  template <class Scalar>
  static Eigen::Matrix<Scalar, 4, 1> bearingVectorInWorldJacobian(
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_WBj,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBm,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBa,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_BC,
          const Scalar* parameters) {

  }

  // Generalization of the addition operation,
  //
  //   x_plus_delta = Plus(x, delta)
  //
  // with the condition that Plus(x, 0) = x.
  bool Plus(const double* x, const double* delta, double* x_plus_delta) const {
    return plus(x, delta, x_plus_delta);
  }

  // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
  //
  // jacobian is a row-major GlobalSize() x LocalSize() matrix.
  bool ComputeJacobian(const double* /*x*/, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, kGlobalDim, kLocalDim, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }

  // Size of x.
  int GlobalSize() const final {
    return kGlobalDim;
  }

  // Size of delta.
  int LocalSize() const final {
    return kLocalDim;
  }

  /// \brief Generalization of the addition operation,
  ///        x_plus_delta = Plus(x, delta)
  ///        with the condition that Plus(x, 0) = x.
  /// @param[in] x Variable.
  /// @param[in] delta Perturbation.
  /// @param[out] x_plus_delta Perturbed x.
  static bool plus(const double* x, const double* delta, double* x_plus_delta) {
    Eigen::Map<const Eigen::Vector3d> _delta(delta);
    LWF::ParallaxAnglePoint pap(x[3], x[0], x[1], x[2], x[4], x[5]);
    pap.boxPlus(_delta, pap);

    const double* bearingData = pap.n_.data();
    x_plus_delta[0] = bearingData[0];
    x_plus_delta[1] = bearingData[1];
    x_plus_delta[2] = bearingData[2];
    x_plus_delta[3] = bearingData[3];
    const double* thetaData = pap.theta_.data();
    x_plus_delta[4] = thetaData[0];
    x_plus_delta[5] = thetaData[1];
    return true;
  }

  /// \brief Computes the Jacobian from minimal space to naively overparameterised space as used by ceres.
  /// @param[in] x Variable.
  /// @param[out] jacobian the Jacobian (dimension minDim x dim).
  /// \return True on success.
  static bool liftJacobian(const double* /*x*/, double* jacobian) {
    Eigen::Map<Eigen::Matrix<double, kLocalDim, kGlobalDim, Eigen::RowMajor>> j(jacobian);
    j.setIdentity();
    return true;
  }
};

// add the model switch functions
} // namespace msckf

#endif // INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_
