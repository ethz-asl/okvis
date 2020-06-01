/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Aug 21, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file ThreadedKFVio.cpp
 * @brief Source file for the ThreadedKFVio class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <map>

#include <glog/logging.h>

#include <okvis/ThreadedKFVio.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/KeyframeForLoopDetection.hpp>
#include <msckf/GeneralEstimator.hpp>
#include <msckf/PriorlessEstimator.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {

static const int max_camera_input_queue_size = 10;

// overlap of imu data before and after two consecutive frames [seconds] if too
// large, frame consumer loop will be blocked for too long by waiting for imu
// meas.
// The frontend waits until frame time + lastOptimizedImageDelay_ +
// temporal_imu_data_overlap for each frame.
// When inertial data for a feature in the most recent frame are requested, the
// feature's observation time may exceed the latest available IMU data, so
// temporal_imu_data_overlap should be greater than frame readout time.
static const okvis::Duration temporal_imu_data_overlap(0.02);

#ifdef USE_MOCK
// Constructor for gmock.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters, okvis::MockVioBackendInterface& estimator,
    okvis::MockVioFrontendInterface& frontend)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(estimator),
      frontend_(frontend),
      parameters_(parameters),
      viewerNamePrefix_("Feature matches for camera"),
      maxImuInputQueueSize_(60) {
  init();
}
#else
// Constructor.
ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      frontend_(new okvis::Frontend(parameters.nCameraSystem.numCameras(),
                                    parameters.frontendOptions)),
      loopClosureModule_(),
      parameters_(parameters),
      viewerNamePrefix_("Feature matches for camera"),
      maxImuInputQueueSize_(2 * max_camera_input_queue_size *
                            parameters.imu.rate /
                            parameters.sensors_information.cameraRate) {
  switch (parameters.optimization.algorithm) {
    case EstimatorAlgorithm::General:
      estimator_.reset(new okvis::GeneralEstimator());
      break;
    case EstimatorAlgorithm::Priorless:
      estimator_.reset(new okvis::PriorlessEstimator());
      break;
    case EstimatorAlgorithm::OKVIS:
    default:
      estimator_.reset(new okvis::Estimator());
      break;
  }
  configureBackendAndFrontendPartly(parameters);

  setBlocking(false);
  init();
}

ThreadedKFVio::ThreadedKFVio(okvis::VioParameters& parameters,
                             std::shared_ptr<Estimator> estimator,
                             std::shared_ptr<okvis::Frontend> frontend, 
                             std::shared_ptr<okvis::LoopClosureMethod> loopClosureMethod)
    : speedAndBiases_propagated_(okvis::SpeedAndBias::Zero()),
      imu_params_(parameters.imu),
      repropagationNeeded_(false),
      frameSynchronizer_(okvis::FrameSynchronizer(parameters)),
      lastAddedImageTimestamp_(okvis::Time(0, 0)),
      optimizationDone_(true),
      estimator_(estimator),
      frontend_(frontend),
      loopClosureModule_(loopClosureMethod),
      parameters_(parameters),
      viewerNamePrefix_("Feature matches for camera"),
      maxImuInputQueueSize_(
          2 * max_camera_input_queue_size * parameters.imu.rate
              / parameters.sensors_information.cameraRate) {
  configureBackendAndFrontendPartly(parameters);
  setBlocking(false);
  init();
}
#endif

// Initialises settings and calls startThreads().
void ThreadedKFVio::init() {
  assert(parameters_.nCameraSystem.numCameras() > 0);
  numCameras_ = parameters_.nCameraSystem.numCameras();
  numCameraPairs_ = 1;

  frontend_->setBriskDetectionOctaves(parameters_.optimization.detectionOctaves);
  frontend_->setBriskDetectionThreshold(parameters_.optimization.detectionThreshold);
  frontend_->setBriskDetectionMaximumKeypoints(parameters_.optimization.maxNoKeypoints);

  frontend_->setKeyframeInsertionOverlapThreshold(parameters_.optimization.keyframeInsertionOverlapThreshold);
  frontend_->setKeyframeInsertionMatchingRatioThreshold(parameters_.optimization.keyframeInsertionMatchingRatioThreshold);
  std::cout <<"Resetting overlap and matching ratio threshold "<< frontend_->getKeyframeInsertionOverlapThershold() <<" "<<
              frontend_->getKeyframeInsertionMatchingRatioThreshold()<<std::endl;

  lastOptimizedImageDelay_ = okvis::Duration(parameters_.nCameraSystem.cameraGeometry(0)->imageDelay());
  lastOptimizedStateTimestamp_ = okvis::Time(0.0) + Estimator::half_window_;;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)
  lastAddedStateTimestamp_ = okvis::Time(0.0) + Estimator::half_window_;  // s.t. last_timestamp_ - overlap >= 0 (since okvis::time(-0.02) returns big number)

  estimator_->addImu(parameters_.imu);
  estimator_->addCameraSystem(parameters_.nCameraSystem);
  for (size_t i = 0; i < numCameras_; ++i) {
    // parameters_.camera_extrinsics is never set (default 0's)...
    // do they ever change?
    estimator_->addCameraParameterStds(parameters_.camera_extrinsics);
    cameraMeasurementsReceived_.emplace_back(
          std::shared_ptr<threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> > >
          (new threadsafe::ThreadSafeQueue<std::shared_ptr<okvis::CameraMeasurement> >()));
  }
  
  // set up windows so things don't crash on Mac OS
  if(parameters_.visualization.displayImages){
    for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
      std::stringstream windowname;
      windowname << viewerNamePrefix_ << " " << im;
  	  cv::namedWindow(windowname.str());
    }
  }
  startThreads();
}

// Start all threads.
void ThreadedKFVio::startThreads() {

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.emplace_back(&ThreadedKFVio::frameConsumerLoop, this, i);
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.emplace_back(&ThreadedKFVio::matchingLoop, this);
  }
  imuConsumerThread_ = std::thread(&ThreadedKFVio::imuConsumerLoop, this);
  positionConsumerThread_ = std::thread(&ThreadedKFVio::positionConsumerLoop,
                                        this);
  gpsConsumerThread_ = std::thread(&ThreadedKFVio::gpsConsumerLoop, this);
  magnetometerConsumerThread_ = std::thread(
      &ThreadedKFVio::magnetometerConsumerLoop, this);
  differentialConsumerThread_ = std::thread(
      &ThreadedKFVio::differentialConsumerLoop, this);

  // algorithm threads
  visualizationThread_ = std::thread(&ThreadedKFVio::visualizationLoop, this);
  optimizationThread_ = std::thread(&ThreadedKFVio::optimizationLoop, this);
  publisherThread_ = std::thread(&ThreadedKFVio::publisherLoop, this);
  loopClosureModule_.startThreads();
}

// Destructor. This calls Shutdown() for all threadsafe queues and joins all threads.
ThreadedKFVio::~ThreadedKFVio() {
  loopClosureModule_.shutdown();
  loopFrames_.Shutdown();

  for (size_t i = 0; i < numCameras_; ++i) {
    cameraMeasurementsReceived_.at(i)->Shutdown();
  }
  keypointMeasurements_.Shutdown();
  matchedFrames_.Shutdown();
  imuMeasurementsReceived_.Shutdown();
  optimizationResults_.Shutdown();
  visualizationData_.Shutdown();
  imuFrameSynchronizer_.shutdown();
  positionMeasurementsReceived_.Shutdown();

  // consumer threads
  for (size_t i = 0; i < numCameras_; ++i) {
    frameConsumerThreads_.at(i).join();
  }
  for (size_t i = 0; i < numCameraPairs_; ++i) {
    keypointConsumerThreads_.at(i).join();
  }
  imuConsumerThread_.join();
  positionConsumerThread_.join();
  gpsConsumerThread_.join();
  magnetometerConsumerThread_.join();
  differentialConsumerThread_.join();
  visualizationThread_.join();
  optimizationThread_.join();
  publisherThread_.join();

  /*okvis::kinematics::Transformation endPosition;
  estimator_->get_T_WS(estimator_->currentFrameId(), endPosition);
  std::stringstream s;
  s << endPosition.r();
  LOG(INFO) << "Sensor end position:\n" << s.str();
  LOG(INFO) << "Distance to origin: " << endPosition.r().norm();*/
#ifndef DEACTIVATE_TIMERS
  LOG(INFO) << okvis::timing::Timing::print();
#endif
}

// Add a new image.
bool ThreadedKFVio::addImage(const okvis::Time & stamp, size_t cameraIndex,
                             const cv::Mat & image,
                             const std::vector<cv::KeyPoint> * keypoints,
                             int frameIdInSource,
                             bool* /*asKeyframe*/) {
  assert(cameraIndex<numCameras_);

  if (lastAddedImageTimestamp_ > stamp
      && fabs((lastAddedImageTimestamp_ - stamp).toSec())
          > parameters_.sensors_information.frameTimestampTolerance) {
    LOG(ERROR)
        << "Received image from the past. Dropping the image.";
    return false;
  }
  lastAddedImageTimestamp_ = stamp;

  std::shared_ptr<okvis::CameraMeasurement> frame = std::make_shared<
      okvis::CameraMeasurement>();
  frame->measurement.image = image;
  frame->measurement.idInSource = frameIdInSource;
  frame->timeStamp = stamp;
  frame->sensorId = cameraIndex;

  if (keypoints != nullptr) {
    frame->measurement.deliversKeypoints = true;
    frame->measurement.keypoints = *keypoints;
  } else {
    frame->measurement.deliversKeypoints = false;
  }

  if (blocking_) {
    cameraMeasurementsReceived_[cameraIndex]->PushBlockingIfFull(frame, 1);
    return true;
  } else {
    cameraMeasurementsReceived_[cameraIndex]->PushNonBlockingDroppingIfFull(
        frame, max_camera_input_queue_size);
    size_t measSize = cameraMeasurementsReceived_[cameraIndex]->Size();
    if (measSize * 2 > max_camera_input_queue_size) {
      LOG(WARNING) << "Exceptional camera meas size " << measSize;
    }
    return measSize == 1;
  }
}

// Add an abstracted image observation.
bool ThreadedKFVio::addKeypoints(
    const okvis::Time & /*stamp*/, size_t /*cameraIndex*/,
    const std::vector<cv::KeyPoint> & /*keypoints*/,
    const std::vector<uint64_t> & /*landmarkIds*/,
    const cv::Mat & /*descriptors*/,
    bool* /*asKeyframe*/) {
  OKVIS_THROW(
      Exception,
      "ThreadedKFVio::addKeypoints() not implemented anymore since changes to _keypointMeasurements queue.");
  return false;
}

// Add an IMU measurement.
bool ThreadedKFVio::addImuMeasurement(const okvis::Time & stamp,
                                      const Eigen::Vector3d & alpha,
                                      const Eigen::Vector3d & omega) {

  okvis::ImuMeasurement imu_measurement;
  imu_measurement.measurement.accelerometers = alpha;
  imu_measurement.measurement.gyroscopes = omega;
  imu_measurement.timeStamp = stamp;

  if (blocking_) {
    imuMeasurementsReceived_.PushBlockingIfFull(imu_measurement, 1);
    return true;
  } else {
    imuMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        imu_measurement, maxImuInputQueueSize_);
    return imuMeasurementsReceived_.Size() == 1;
  }
}

// Add a position measurement.
void ThreadedKFVio::addPositionMeasurement(const okvis::Time & stamp,
                                           const Eigen::Vector3d & position,
                                           const Eigen::Vector3d & positionOffset,
                                           const Eigen::Matrix3d & positionCovariance) {
  okvis::PositionMeasurement position_measurement;
  position_measurement.measurement.position = position;
  position_measurement.measurement.positionOffset = positionOffset;
  position_measurement.measurement.positionCovariance = positionCovariance;
  position_measurement.timeStamp = stamp;

  if (blocking_) {
    positionMeasurementsReceived_.PushBlockingIfFull(position_measurement, 1);
    return;
  } else {
    positionMeasurementsReceived_.PushNonBlockingDroppingIfFull(
        position_measurement, maxPositionInputQueueSize_);
    return;
  }
}

// Add a GPS measurement.
void ThreadedKFVio::addGpsMeasurement(const okvis::Time &, double, double,
                                      double, const Eigen::Vector3d &,
                                      const Eigen::Matrix3d &) {
  OKVIS_THROW(Exception, "GPS measurements not supported")
}

// Add a magnetometer measurement.
void ThreadedKFVio::addMagnetometerMeasurement(const okvis::Time &,
                                               const Eigen::Vector3d &, double) {
  OKVIS_THROW(Exception, "Magnetometer measurements not supported")
}

// Add a static pressure measurement.
void ThreadedKFVio::addBarometerMeasurement(const okvis::Time &, double, double) {

  OKVIS_THROW(Exception, "Barometer measurements not supported")
}

// Add a differential pressure measurement.
void ThreadedKFVio::addDifferentialPressureMeasurement(const okvis::Time &,
                                                       double, double) {

  OKVIS_THROW(Exception, "Differential pressure measurements not supported")
}

// Set the blocking variable that indicates whether the addMeasurement() functions
// should return immediately (blocking=false), or only when the processing is complete.
void ThreadedKFVio::setBlocking(bool blocking) {
  blocking_ = blocking;
  loopClosureModule_.setBlocking(blocking);
  // disable time limit for optimization
  if(blocking_) {
    std::lock_guard<std::mutex> lock(estimator_mutex_);
    estimator_->setOptimizationTimeLimit(-1.0,parameters_.optimization.max_iterations);
  }
}

// Loop to process frames from camera with index cameraIndex
void ThreadedKFVio::frameConsumerLoop(size_t cameraIndex) {
  std::shared_ptr<okvis::CameraMeasurement> frame;
  std::shared_ptr<okvis::MultiFrame> multiFrame;
  TimerSwitchable beforeDetectTimer("1.1 frameLoopBeforeDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer("1.1.1 waitForFrameSynchronizerMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable addNewFrameToSynchronizerTimer("1.1.2 addNewFrameToSynchronizer"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForStateVariablesMutexTimer("1.1.3 waitForStateVariablesMutex"+std::to_string(cameraIndex),true);
  TimerSwitchable propagationTimer("1.1.4 propagationTimer"+std::to_string(cameraIndex),true);
  TimerSwitchable detectTimer("1.2 detectAndDescribe"+std::to_string(cameraIndex),true);
  TimerSwitchable afterDetectTimer("1.3 afterDetect"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForFrameSynchronizerMutexTimer2("1.3.1 waitForFrameSynchronizerMutex2"+std::to_string(cameraIndex),true);
  TimerSwitchable waitForMatchingThreadTimer("1.4 waitForMatchingThread"+std::to_string(cameraIndex),true);


  for (;;) {
    // get data and check for termination request
    if (cameraMeasurementsReceived_[cameraIndex]->PopBlocking(&frame) == false) {
      return;
    }
    beforeDetectTimer.start();
    {  // lock the frame synchronizer
      waitForFrameSynchronizerMutexTimer.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer.stop();
      // add new frame to frame synchronizer and get the MultiFrame containing it
      addNewFrameToSynchronizerTimer.start();
      multiFrame = frameSynchronizer_.addNewFrame(frame);
      addNewFrameToSynchronizerTimer.stop();
    }  // unlock frameSynchronizer only now as we can be sure that not two states are added for the same timestamp
    okvis::kinematics::Transformation T_WS;
    okvis::Time lastTimestamp;
    okvis::SpeedAndBias speedAndBiases;
    okvis::Duration lastImageDelay;
    // copy last state variables
    {
      waitForStateVariablesMutexTimer.start();
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      waitForStateVariablesMutexTimer.stop();
      T_WS = lastOptimized_T_WS_;
      speedAndBiases = lastOptimizedSpeedAndBiases_;
      lastTimestamp = lastOptimizedStateTimestamp_;
      lastImageDelay = lastOptimizedImageDelay_;
    }

    // -- get relevant imu messages for new state
    okvis::Time imuDataEndTime =
        multiFrame->timestamp() + lastImageDelay + temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime = lastTimestamp - Estimator::half_window_;

    OKVIS_ASSERT_TRUE_DBG(Exception,imuDataBeginTime < imuDataEndTime,"imu data end time is smaller than begin time.");

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
      okvis::Time(imuDataEndTime)) == false)  {
      return;
    }
    OKVIS_ASSERT_TRUE_DBG(Exception,
                          imuDataEndTime < imuMeasurements_.back().timeStamp,
                          "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurements(
        imuDataBeginTime, imuDataEndTime, imuMeasurements_, &imuMeasurements_mutex_);

    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0) {
      beforeDetectTimer.stop();
      continue;
    }

    if (imuData.front().timeStamp > frame->timeStamp) {
      LOG(WARNING) << "Frame is newer than oldest IMU measurement. Dropping it.";
      beforeDetectTimer.stop();
      continue;
    }

    // get T_WC(camIndx) for detectAndDescribe()
    if (estimator_->numFrames() == 0) {
      // first frame ever
      bool success = okvis::Estimator::initPoseFromImu(imuData, T_WS);
      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = T_WS;
        lastOptimizedSpeedAndBiases_.head<3>() = parameters_.initialState.v_WS;
        lastOptimizedSpeedAndBiases_.segment<3>(3) = imu_params_.g0;
        lastOptimizedSpeedAndBiases_.segment<3>(6) = imu_params_.a0;
        lastOptimizedImageDelay_ = okvis::Duration(parameters_.nCameraSystem.cameraGeometry(0)->imageDelay());
        lastOptimizedStateTimestamp_ = multiFrame->timestamp() + lastOptimizedImageDelay_;
      }
      OKVIS_ASSERT_TRUE_DBG(Exception, success,
          "pose could not be initialized from imu measurements.");
      if (!success) {
        beforeDetectTimer.stop();
        continue;
      }
    } else {
      // get old T_WS
      propagationTimer.start();
      okvis::ceres::ImuError::propagation(imuData, parameters_.imu, T_WS,
                                          speedAndBiases, lastTimestamp,
                                          multiFrame->timestamp() + lastImageDelay);
      propagationTimer.stop();
    }
    okvis::kinematics::Transformation T_WC = T_WS
        * (*parameters_.nCameraSystem.T_SC(frame->sensorId));
    beforeDetectTimer.stop();
    if (frontend_->isDescriptorBasedMatching()) {
      detectTimer.start();
      frontend_->detectAndDescribe(frame->sensorId, multiFrame, T_WC, nullptr);
      detectTimer.stop();
    }
    afterDetectTimer.start();

    bool push = false;
    {  // we now tell frame synchronizer that detectAndDescribe is done for MF with our timestamp
      waitForFrameSynchronizerMutexTimer2.start();
      std::lock_guard<std::mutex> lock(frameSynchronizer_mutex_);
      waitForFrameSynchronizerMutexTimer2.stop();
      frameSynchronizer_.detectionEndedForMultiFrame(multiFrame->id());

      if (frameSynchronizer_.detectionCompletedForAllCameras(
          multiFrame->id())) {
//        LOG(INFO) << "detection completed for multiframe with id "<< multi_frame->id();
        push = true;
      }
    }  // unlocking frame synchronizer
    afterDetectTimer.stop();
    if (push) {
      // use queue size 1 to propagate a congestion to the _cameraMeasurementsReceived queue
      // and check for termination request
      waitForMatchingThreadTimer.start();
      if (keypointMeasurements_.PushBlockingIfFull(multiFrame, 1) == false) {
        waitForMatchingThreadTimer.stop();
        return;
      }
      waitForMatchingThreadTimer.stop();
    }
  }
}

// Loop that matches frames with existing frames.
void ThreadedKFVio::matchingLoop() {
  TimerSwitchable prepareToAddStateTimer("2.1 prepareToAddState",true);
  TimerSwitchable waitForOptimizationTimer("2.2 waitForOptimization",true);
  TimerSwitchable addStateTimer("2.3 addState",true);
  TimerSwitchable matchingTimer("2.4 matching",true);

  for (;;) {
    // get new frame
    std::shared_ptr<okvis::MultiFrame> frame;

    // get data and check for termination request
    if (keypointMeasurements_.PopBlocking(&frame) == false)
      return;

    prepareToAddStateTimer.start();
    // -- get relevant imu messages for new state
    okvis::Duration lastImageDelay;
    {
      std::lock_guard<std::mutex> lock(lastState_mutex_);
      lastImageDelay = lastOptimizedImageDelay_;
    }
    okvis::Time imuDataEndTime = frame->timestamp() + lastImageDelay +
                                 temporal_imu_data_overlap;
    okvis::Time imuDataBeginTime =
        lastAddedStateTimestamp_ - Estimator::half_window_;
    if (imuDataBeginTime.toSec() == 0.0) {  // first state not yet added
      imuDataBeginTime = frame->timestamp() - Estimator::half_window_;
    }
    // at maximum Duration(.) sec of data is allowed
    if (imuDataEndTime - imuDataBeginTime > Duration(8)) {
      LOG(WARNING) << "Warn: Too long interval between two frames "
                   << lastAddedStateTimestamp_.toSec() << " and "
                   << frame->timestamp().toSec();
      imuDataBeginTime = imuDataEndTime - Duration(8);
    }
    OKVIS_ASSERT_TRUE_DBG(Exception, imuDataBeginTime < imuDataEndTime,
                          "imu data end time is smaller than begin time."
                              << "current frametimestamp " << frame->timestamp()
                              << " (id: " << frame->id() << "last timestamp "
                              << lastAddedStateTimestamp_
                              << " (id: " << estimator_->currentFrameId());

    // wait until all relevant imu messages have arrived and check for termination request
    if (imuFrameSynchronizer_.waitForUpToDateImuData(
        okvis::Time(imuDataEndTime)) == false)
      return; OKVIS_ASSERT_TRUE_DBG(Exception,
        imuDataEndTime < imuMeasurements_.back().timeStamp,
        "Waiting for up to date imu data seems to have failed!");

    okvis::ImuMeasurementDeque imuData = getImuMeasurements(
        imuDataBeginTime, imuDataEndTime, imuMeasurements_, &imuMeasurements_mutex_);

    prepareToAddStateTimer.stop();
    // if imu_data is empty, either end_time > begin_time or
    // no measurements in timeframe, should not happen, as we waited for measurements
    if (imuData.size() == 0)
      continue;

    // make sure that optimization of last frame is over.
    // TODO If we didn't actually 'pop' the _matchedFrames queue until after optimization this would not be necessary
    {
      waitForOptimizationTimer.start();
      std::unique_lock<std::mutex> l(estimator_mutex_);
      while (!optimizationDone_)
        optimizationNotification_.wait(l);
      waitForOptimizationTimer.stop();
      addStateTimer.start();
      okvis::Time t0Matching = okvis::Time::now();
      bool asKeyframe = false;
      if (estimator_->addStates(frame, imuData, asKeyframe)) {
        lastAddedStateTimestamp_ = frame->timestamp() + lastImageDelay;
        addStateTimer.stop();
      } else {
        LOG(ERROR) << "Failed to add state! will drop multiframe.";
        addStateTimer.stop();
        continue;
      }

      // -- matching keypoints, initialising landmarks etc.
      okvis::kinematics::Transformation T_WS;
      estimator_->get_T_WS(frame->id(), T_WS);
      matchingTimer.start();
      frontend_->dataAssociationAndInitialization(*estimator_, T_WS, parameters_,
                                                 map_, frame, &asKeyframe);
      matchingTimer.stop();
      if (asKeyframe)
        estimator_->setKeyframe(frame->id(), asKeyframe);
      if(!blocking_) {
        double timeLimit = parameters_.optimization.timeLimitForMatchingAndOptimization
                           -(okvis::Time::now()-t0Matching).toSec();
        estimator_->setOptimizationTimeLimit(std::max<double>(0.0, timeLimit),
                                            parameters_.optimization.min_iterations);
      }
      optimizationDone_ = false;
    }  // unlock estimator_mutex_

    // use queue size 1 to propagate a congestion to the _matchedFrames queue
    if (matchedFrames_.PushBlockingIfFull(frame, 1) == false)
      return;
  }
}

// Loop to process IMU measurements.
void ThreadedKFVio::imuConsumerLoop() {
  okvis::ImuMeasurement data;
  TimerSwitchable processImuTimer("0 processImuMeasurements",true);
  for (;;) {
    // get data and check for termination request
    if (imuMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    processImuTimer.start();
    okvis::Time start;
    const okvis::Time* end;  // do not need to copy end timestamp
    {
      std::lock_guard<std::mutex> imuLock(imuMeasurements_mutex_);
      OKVIS_ASSERT_TRUE(Exception,
                        imuMeasurements_.empty()
                        || imuMeasurements_.back().timeStamp < data.timeStamp,
                        "IMU measurement from the past received");

      if (parameters_.publishing.publishImuPropagatedState) {
        if (!repropagationNeeded_ && imuMeasurements_.size() > 0) {
          start = imuMeasurements_.back().timeStamp;
        } else if (repropagationNeeded_) {
          std::lock_guard<std::mutex> lastStateLock(lastState_mutex_);
          start = lastOptimizedStateTimestamp_;
          T_WS_propagated_ = lastOptimized_T_WS_;
          speedAndBiases_propagated_ = lastOptimizedSpeedAndBiases_;
          repropagationNeeded_ = false;
        } else
          start = okvis::Time(0, 0);
        end = &data.timeStamp;
      }
      imuMeasurements_.push_back(data);
    }  // unlock _imuMeasurements_mutex

    // notify other threads that imu data with timeStamp is here.
    imuFrameSynchronizer_.gotImuData(data.timeStamp);

    if (parameters_.publishing.publishImuPropagatedState) {
      Eigen::Matrix<double, 15, 15> covariance;
      Eigen::Matrix<double, 15, 15> jacobian;

      frontend_->propagation(imuMeasurements_, imu_params_, T_WS_propagated_,
                            speedAndBiases_propagated_, start, *end, &covariance,
                            &jacobian);
      OptimizationResults result;
      result.stamp = *end;
      result.T_WS = T_WS_propagated_;
      result.speedAndBiases = speedAndBiases_propagated_;
      result.omega_S = imuMeasurements_.back().measurement.gyroscopes
          - speedAndBiases_propagated_.segment<3>(3);
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        result.vector_of_T_SCi.push_back(
            okvis::kinematics::Transformation(
                *parameters_.nCameraSystem.T_SC(i)));
      }
      result.onlyPublishLandmarks = false;
      optimizationResults_.PushNonBlockingDroppingIfFull(result,1);
    }
    processImuTimer.stop();
  }
}

// Loop to process position measurements.
void ThreadedKFVio::positionConsumerLoop() {
  okvis::PositionMeasurement data;
  for (;;) {
    // get data and check for termination request
    if (positionMeasurementsReceived_.PopBlocking(&data) == false)
      return;
    // collect
    {
      std::lock_guard<std::mutex> positionLock(positionMeasurements_mutex_);
      positionMeasurements_.push_back(data);
    }
  }
}

// Loop to process GPS measurements.
void ThreadedKFVio::gpsConsumerLoop() {
}

// Loop to process magnetometer measurements.
void ThreadedKFVio::magnetometerConsumerLoop() {
}

// Loop to process differential pressure measurements.
void ThreadedKFVio::differentialConsumerLoop() {
}

// Loop that visualizes completed frames.
void ThreadedKFVio::visualizationLoop() {
  okvis::VioVisualizer visualizer_(parameters_, viewerNamePrefix_);
  for (;;) {
    VioVisualizer::VisualizationData::Ptr new_data;
    if (visualizationData_.PopBlocking(&new_data) == false)
      return;
    //visualizer_.showDebugImages(new_data);
    const bool drawMatches = true;
    std::vector<cv::Mat> out_images(parameters_.nCameraSystem.numCameras());
    if (drawMatches) {
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        out_images[i] = visualizer_.drawMatches(new_data, i);
      }
    } else {
      for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
        out_images[i] = visualizer_.drawColoredKeypoints(new_data, i);
      }
    }
    displayImages_.PushNonBlockingDroppingIfFull(out_images,1);
  }
}

// trigger display (needed because OSX won't allow threaded display)
void ThreadedKFVio::display() {
  std::vector<cv::Mat> out_images;
  if (displayImages_.Size() == 0)
	return;
  if (displayImages_.PopBlocking(&out_images) == false)
    return;
  // draw
  for (size_t im = 0; im < parameters_.nCameraSystem.numCameras(); im++) {
    std::stringstream windowname;
    windowname << viewerNamePrefix_ << " " << im;
    cv::imshow(windowname.str(), out_images[im]);
  }
  cv::waitKey(1);
}

// Loop that performs the optimization and marginalisation.
void ThreadedKFVio::optimizationLoop() {
  TimerSwitchable optimizationTimer("3.1 optimization",true);
  TimerSwitchable marginalizationTimer("3.2 marginalization",true);
  TimerSwitchable afterOptimizationTimer("3.3 afterOptimization",true);

  for (;;) {
    std::shared_ptr<okvis::MultiFrame> frame_pairs;
    VioVisualizer::VisualizationData::Ptr visualizationDataPtr;
    okvis::Time deleteImuMeasurementsUntil(0, 0);
    std::vector<std::shared_ptr<LoopFrameAndMatches>> loopFrameAndMatchesList;
    bool foundLoop = popLoopFrameAndMatchesList(&loopFrameAndMatchesList);
    if (matchedFrames_.PopBlocking(&frame_pairs) == false)
      return;
    OptimizationResults result;
    std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe;
    {
      std::lock_guard<std::mutex> l(estimator_mutex_);
      optimizationTimer.start();
      if (foundLoop) {
        estimator_->setLoopFrameAndMatchesList(loopFrameAndMatchesList);
      }
      //if(frontend_->isInitialized()){
      estimator_->optimize(parameters_.optimization.max_iterations, 2, false);
      //}
      /*if (estimator_->numFrames() > 0 && !frontend_->isInitialized()){
        // undo translation
        for(size_t n=0; n<estimator_->numFrames(); ++n){
          okvis::kinematics::Transformation T_WS_0;
          estimator_->get_T_WS(estimator_->frameIdByAge(n),T_WS_0);
          Eigen::Matrix4d T_WS_0_mat = T_WS_0.T();
          T_WS_0_mat.topRightCorner<3,1>().setZero();
          estimator_->set_T_WS(estimator_->frameIdByAge(n),okvis::kinematics::Transformation(T_WS_0_mat));
          okvis::SpeedAndBias sb_0 = okvis::SpeedAndBias::Zero();
          if(estimator_->getSpeedAndBias(estimator_->frameIdByAge(n), 0, sb_0)){
            sb_0.head<3>().setZero();
            estimator_->setSpeedAndBias(estimator_->frameIdByAge(n), 0, sb_0);
          }
        }
      }*/

      optimizationTimer.stop();

      // get timestamp of last frame in IMU window. Need to do this before marginalization as it will be removed there (if not keyframe)
      /*if (estimator_->numFrames()
          > size_t(parameters_.optimization.numImuFrames)) {
        deleteImuMeasurementsUntil = estimator_->multiFrame(
            estimator_->frameIdByAge(parameters_.optimization.numImuFrames))
            ->timestamp() - temporal_imu_data_overlap;
      }*/
      okvis::Duration optimizedImageDelay;
      estimator_->getImageDelay(frame_pairs->id(), 0, &optimizedImageDelay);
      deleteImuMeasurementsUntil =
          estimator_->oldestFrameTimestamp() + optimizedImageDelay - Estimator::half_window_;

      marginalizationTimer.start();
      estimator_->setKeyframeRedundancyThresholds(
          parameters_.optimization.translationThreshold,
          parameters_.optimization.rotationThreshold,
          parameters_.optimization.trackingRateThreshold,
          parameters_.optimization.minTrackLength,
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames);
      estimator_->applyMarginalizationStrategy(
          parameters_.optimization.numKeyframes,
          parameters_.optimization.numImuFrames,
          result.transferredLandmarks);
      marginalizationTimer.stop();
      afterOptimizationTimer.start();

      // now actually remove measurements
      deleteImuMeasurements(deleteImuMeasurementsUntil,
                            imuMeasurements_, &imuMeasurements_mutex_);

      // saving optimized state and saving it in OptimizationResults struct
      okvis::kinematics::Transformation latest_T_WS;
      okvis::SpeedAndBias latestSpeedAndBias;
      uint64_t latestNFrameId = frame_pairs->id();
      estimator_->get_T_WS(latestNFrameId, latest_T_WS);
      estimator_->getSpeedAndBias(latestNFrameId, 0, latestSpeedAndBias);
      okvis::Time latestStateTime = estimator_->currentFrameTimestamp();

      {
        std::lock_guard<std::mutex> lock(lastState_mutex_);
        lastOptimized_T_WS_ = latest_T_WS;
        lastOptimizedSpeedAndBiases_ = latestSpeedAndBias;
        lastOptimizedImageDelay_ = optimizedImageDelay;
        lastOptimizedStateTimestamp_ = latestStateTime;
      }
      {
        // if we publish the state after each IMU propagation we do not need to publish it here.
        if (!parameters_.publishing.publishImuPropagatedState) {
          result.T_WS = latest_T_WS;
          result.speedAndBiases = latestSpeedAndBias;
          result.stamp = latestStateTime;
          result.onlyPublishLandmarks = false;
        } else {
          result.onlyPublishLandmarks = true;
        }
        estimator_->getLandmarks(result.landmarksVector);
        dumpCalibrationParameters(latestNFrameId, &result);
        estimator_->getLoopQueryKeyframeMessage(frame_pairs, &queryKeyframe);

        repropagationNeeded_ = true;
      }

      if (parameters_.visualization.displayImages) {
        // fill in information that requires access to estimator.
        visualizationDataPtr = VioVisualizer::VisualizationData::Ptr(
            new VioVisualizer::VisualizationData());
        visualizationDataPtr->observations.resize(frame_pairs->numKeypoints());
        okvis::MapPoint landmark;
        okvis::ObservationVector::iterator it = visualizationDataPtr
            ->observations.begin();
        for (size_t camIndex = 0; camIndex < frame_pairs->numFrames();
            ++camIndex) {
          for (size_t k = 0; k < frame_pairs->numKeypoints(camIndex); ++k) {
            OKVIS_ASSERT_TRUE_DBG(Exception,it != visualizationDataPtr->observations.end(),"Observation-vector not big enough");
            it->keypointIdx = k;
            frame_pairs->getKeypoint(camIndex, k, it->keypointMeasurement);
            frame_pairs->getKeypointSize(camIndex, k, it->keypointSize);
            it->cameraIdx = camIndex;
            it->frameId = frame_pairs->id();
            it->landmarkId = frame_pairs->landmarkId(camIndex, k);
            if (estimator_->isLandmarkAdded(it->landmarkId)) {
              estimator_->getLandmark(it->landmarkId, landmark);
              it->landmark_W = landmark.pointHomog;
              it->numObservations = static_cast<int>(landmark.observations.size());
              if (estimator_->isLandmarkInitialized(it->landmarkId))
                it->isInitialized = true;
              else
                it->isInitialized = false;
            } else {
              it->landmark_W = Eigen::Vector4d(0, 0, 0, 0);  // set to infinity to tell visualizer that landmark is not added
            }
            ++it;
          }
        }
        visualizationDataPtr->keyFrames = estimator_->multiFrame(
            estimator_->currentKeyframeId());
        estimator_->get_T_WS(estimator_->currentKeyframeId(),
                            visualizationDataPtr->T_WS_keyFrame);
      }
      afterOptimizationTimer.stop();
      optimizationDone_ = true;
    }  // unlock mutex
    optimizationNotification_.notify_all();

    optimizationResults_.Push(result);
    loopClosureModule_.push(queryKeyframe);
    // adding further elements to visualization data that do not access estimator
    if (parameters_.visualization.displayImages) {
      visualizationDataPtr->currentFrames = frame_pairs;
      visualizationData_.PushNonBlockingDroppingIfFull(visualizationDataPtr, 1);
    }
  }
}

void ThreadedKFVio::dumpCalibrationParameters(uint64_t latestNFrameId, OptimizationResults* result) const {
  int frameIdInSource = -1;
  bool isKF= false;
  estimator_->getFrameId(latestNFrameId, frameIdInSource, isKF);
  result->frameIdInSource = frameIdInSource;
  result->isKeyframe = isKF;

  result->vector_of_T_SCi.clear();
  result->opt_T_SCi_coeffs.clear();
  for (size_t i = 0; i < parameters_.nCameraSystem.numCameras(); ++i) {
    okvis::kinematics::Transformation T_SC;
    estimator_->getCameraSensorStates(latestNFrameId, i, T_SC);
    result->vector_of_T_SCi.emplace_back(T_SC);

    int extrinsic_opt_type = estimator_->getCameraExtrinsicOptType(i);
    Eigen::VectorXd optimized_coeffs;
    ExtrinsicModelToParamValues(extrinsic_opt_type, T_SC, &optimized_coeffs);
    result->opt_T_SCi_coeffs.emplace_back(optimized_coeffs);
  }

  estimator_->getImuAugmentedStatesEstimate(&result->imuExtraParams_);
  estimator_->getCameraCalibrationEstimate(&result->cameraParams_);

  // bool covStatus = estimator_->getStateVariance(&result->stateVariance_);
}

// Loop that publishes the newest state and landmarks.
void ThreadedKFVio::publisherLoop() {
  for (;;) {
    // get the result data
    OptimizationResults result;
    if (optimizationResults_.PopBlocking(&result) == false)
      return;

    // call all user callbacks
    if (stateCallback_ && !result.onlyPublishLandmarks)
      stateCallback_(result.stamp, result.T_WS);
    if (fullStateCallback_ && !result.onlyPublishLandmarks)
      fullStateCallback_(result.stamp, result.T_WS, result.speedAndBiases,
                         result.omega_S, result.frameIdInSource);
    if (fullStateCallbackWithExtrinsics_ && !result.onlyPublishLandmarks)
      fullStateCallbackWithExtrinsics_(result.stamp, result.T_WS,
                                       result.speedAndBiases, result.omega_S,
                                       result.frameIdInSource,
                                       result.vector_of_T_SCi);
    if (fullStateCallbackWithAllCalibration_) {
      fullStateCallbackWithAllCalibration_(
          result.stamp, result.T_WS, result.speedAndBiases, result.omega_S,
          result.frameIdInSource, result.opt_T_SCi_coeffs, result.imuExtraParams_,
          result.cameraParams_, result.stateVariance_, result.vector_of_T_SCi);
    }
    if (landmarksCallback_ && !result.landmarksVector.empty())
      landmarksCallback_(result.stamp, result.landmarksVector,
                         result.transferredLandmarks);  //TODO(gohlp): why two maps?
  }
}

void ThreadedKFVio::saveStatistics(const std::string &filename) const {
  std::ofstream stream(filename, std::ios_base::app);
  if (!stream.is_open()) {
    LOG(WARNING) << "error in opening " << filename;
    return;
  }
  estimator_->printTrackLengthHistogram(stream);
  if (stream.is_open())
    stream.close();
}

void ThreadedKFVio::configureBackendAndFrontendPartly(okvis::VioParameters& parameters) {
  frontend_->setLandmarkTriangulationParameters(
      parameters.optimization.triangulationTranslationThreshold,
      parameters.optimization.triangulationMaxDepth);
  estimator_->setInitialNavState(
      InitialNavState(parameters.initialState));
  estimator_->setUseEpipolarConstraint(parameters.optimization.useEpipolarConstraint);
  estimator_->setCameraObservationModel(parameters.optimization.cameraObservationModelId);
  estimator_->setPointLandmarkOptions(parameters.pointLandmarkOptions);
  estimator_->setPoseGraphOptions(parameters.poseGraphOptions);

  loopClosureModule_.setOutputLoopFrameCallback(
      std::bind(&okvis::ThreadedKFVio::addLoopFrameAndMatches,
                this, std::placeholders::_1));
}

bool ThreadedKFVio::popLoopFrameAndMatchesList(
    std::vector<std::shared_ptr<LoopFrameAndMatches>>*
        loopFrameAndMatchesList) {
  std::shared_ptr<LoopFrameAndMatches> loopFrameAndMatches;
  bool foundLoop = loopFrames_.PopNonBlocking(&loopFrameAndMatches);
  while (foundLoop) {
    loopFrameAndMatchesList->push_back(loopFrameAndMatches);
    foundLoop = loopFrames_.PopNonBlocking(&loopFrameAndMatches);
  }
  return loopFrameAndMatchesList->size() > 0u;
}

bool ThreadedKFVio::addLoopFrameAndMatches(std::shared_ptr<LoopFrameAndMatches> loopFrame) {
  if (blocking_) {
    loopFrames_.PushBlockingIfFull(loopFrame, 2); // 2 because loop closure is not critical.
    return true;
  } else {
    loopFrames_.PushNonBlockingDroppingIfFull(
        loopFrame, maxImuInputQueueSize_);
    return loopFrames_.Size() == 1;
  }
}

void ThreadedKFVio::appendPgoStateCallback(const StateCallback& pgoStateCallback) {
  loopClosureModule_.appendStateCallback(pgoStateCallback);
}

}  // namespace okvis
