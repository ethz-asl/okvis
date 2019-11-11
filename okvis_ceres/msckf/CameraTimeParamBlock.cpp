/**
 * @file CameraTimeParamBlock.cpp
 * @brief Source file for the CameraTimeParamBlock class.
 * @author Jianzhu Huai
 */

#include <msckf/CameraTimeParamBlock.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

// Default constructor (assumes not fixed).
CameraTimeParamBlock::CameraTimeParamBlock()
    : base_t::ParameterBlockSized() {
  setFixed(false);
}

// Trivial destructor.
CameraTimeParamBlock::~CameraTimeParamBlock() {
}

// Constructor with estimate and time.
CameraTimeParamBlock::CameraTimeParamBlock(
    const TimeOffset& timeOffset, uint64_t id,
    const okvis::Time& timestamp) {
  setEstimate(timeOffset);
  setId(id);
  setTimestamp(timestamp);
  setFixed(false);
}

// setters
// Set estimate of this parameter block.
void CameraTimeParamBlock::setEstimate(const TimeOffset& timeOffset) {
  for (int i = 0; i < base_t::Dimension; ++i)
    parameters_[i] = timeOffset;
}

// getters
// Get estimate.
TimeOffset CameraTimeParamBlock::estimate() const {
  TimeOffset timeOffset;
  for (int i = 0; i < base_t::Dimension; ++i)
    timeOffset = parameters_[i];
  return timeOffset;
}

}  // namespace ceres
}  // namespace okvis
