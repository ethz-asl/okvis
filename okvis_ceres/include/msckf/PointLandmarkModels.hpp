#ifndef INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_
#define INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace msckf {
// [x, y, z, w] usually expressed in the world frame.
class HomogeneousPointParameterization
{
public:
 static const int kModelId = 0;

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
  template <class Scalar>
  static Eigen::Matrix<Scalar, 4, 1> bearingVectorInWorld(
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_WBj,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBm,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBa,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_BC,
          const Scalar* parameters) {

  }

  template <class Scalar>
  static Eigen::Matrix<Scalar, 4, 1> bearingVectorInCamera(
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
class ParallaxAngleParameterization {
public:
  static const int kModelId = 2;
  template <class Scalar>
  static Eigen::Matrix<Scalar, 4, 1> bearingVectorInWorld(
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_WBj,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBm,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBa,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_BC,
          const Scalar* parameters) {

  }

  template <class Scalar>
  static Eigen::Matrix<Scalar, 4, 1> bearingVectorInCamera(
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_WBj,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBm,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* pairT_WBa,
          const std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>& pairT_BC,
          const Scalar* parameters) {

  }
};

// add the model switch functions
} // namespace msckf

#endif // INCLUDE_MSCKF_POINT_LANDMARK_MODELS_HPP_
