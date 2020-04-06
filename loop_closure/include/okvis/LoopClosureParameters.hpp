/**
 * @file LoopClosureParameters.hpp
 * @brief Header file for LoopClosureParameters class which encompasses
 * parameters for loop detection and pose graph optimization.
 */

#ifndef INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_
#define INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_

#include <Eigen/Core>

namespace okvis {
class LoopClosureParameters {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LoopClosureParameters();
  ~LoopClosureParameters();
  size_t methodId;
};
}  // namespace okvis

#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_
