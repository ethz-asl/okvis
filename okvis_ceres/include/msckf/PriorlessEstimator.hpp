
/**
 * @file msckf/PriorlessEstimator.hpp
 * @brief Header file for the PriorlessEstimator class. This does all the backend work.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_MSCKF_PRIORLESS_ESTIMATOR_HPP_
#define INCLUDE_MSCKF_PRIORLESS_ESTIMATOR_HPP_

#include <memory>
#include <mutex>
#include <array>

#include <ceres/ceres.h>
#include <okvis/kinematics/Transformation.hpp>

#include <okvis/assert_macros.hpp>
#include <okvis/VioBackendInterface.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/Map.hpp>
#include <okvis/ceres/MarginalizationError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/CeresIterationCallback.hpp>
#include <okvis/Estimator.hpp>

#include <msckf/CameraRig.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
class PriorlessEstimator : public Estimator
{
 public:
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief The default constructor.
   */
  PriorlessEstimator();

  /**
   * @brief Constructor if a ceres map is already available.
   * @param mapPtr Shared pointer to ceres map.
   */
  PriorlessEstimator(std::shared_ptr<okvis::ceres::Map> mapPtr);
  virtual ~PriorlessEstimator();

  /**
   * @brief Add a pose to the state.
   * @param multiFrame Matched multiFrame.
   * @param imuMeasurements IMU measurements from last state to new one.
   * @param asKeyframe Is this new frame a keyframe?
   * @return True if successful.
   */
  virtual bool addStates(okvis::MultiFramePtr multiFrame,
                 const okvis::ImuMeasurementDeque & imuMeasurements,
                 bool asKeyframe);



  /**
   * @brief Applies the dropping/marginalization strategy according to the RSS'13/IJRR'14 paper.
   *        The new number of frames in the window will be numKeyframes+numImuFrames.
   * @param numKeyframes Number of keyframes.
   * @param numImuFrames Number of frames in IMU window.
   * @param removedLandmarks Get the landmarks that were removed by this operation.
   * @return True if successful.
   */
  virtual bool applyMarginalizationStrategy(size_t numKeyframes, size_t numImuFrames,
                                    okvis::MapPointVector& removedLandmarks);


  /**
   * @brief Start ceres optimization.
   * @param[in] numIter Maximum number of iterations.
   * @param[in] numThreads Number of threads.
   * @param[in] verbose Print out optimization progress and result, if true.
   */
  virtual void optimize(size_t numIter, size_t numThreads = 1, bool verbose = false);

};
}  // namespace okvis

#endif /* #ifndef INCLUDE_MSCKF_PRIORLESS_ESTIMATOR_HPP_ */
