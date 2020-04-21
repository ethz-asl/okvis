#include <glog/logging.h>

#include <okvis/LoopClosureModule.hpp>

namespace okvis {
LoopClosureModule::LoopClosureModule()
    : blocking_(false), loopClosureMethod_(new LoopClosureMethod()) {}

LoopClosureModule::LoopClosureModule(
    std::shared_ptr<okvis::LoopClosureMethod> loopClosureMethod)
    : blocking_(false), loopClosureMethod_(loopClosureMethod) {}

LoopClosureModule::~LoopClosureModule() {}

void LoopClosureModule::setBlocking(bool blocking) { blocking_ = blocking; }

bool LoopClosureModule::push(
    std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe) {
  if (!queryKeyframe) {  // skip empty packets to save bandwidth.
    return false;
  }
  if (blocking_) {
    queryKeyframeList_.PushBlockingIfFull(queryKeyframe, 1);
    return true;
  } else {
    queryKeyframeList_.PushNonBlockingDroppingIfFull(
        queryKeyframe, maxQueryKeyframeQueueSize_);
    return queryKeyframeList_.Size() == 1;
  }
}

void LoopClosureModule::setOutputLoopFrameCallback(
    const OutputLoopFrameCallback& outputCallback) {
  outputLoopFrameCallback_ = outputCallback;
}

void LoopClosureModule::appendStateCallback(
    const VioInterface::StateCallback& stateCallback) {
  stateCallbackList_.push_back(stateCallback);
}

void LoopClosureModule::startThreads() {
  loopClosureThread_ = std::thread(&LoopClosureModule::loopClosureLoop, this);
  publisherThread_ = std::thread(&LoopClosureModule::publisherLoop, this);
}

void LoopClosureModule::loopClosureLoop() {
  TimerSwitchable loopDetectionTimer("4.1 loopDetection", true);
  TimerSwitchable poseGraphOptTimer("4.2 poseGraphOpt", true);
  for (;;) {
    std::shared_ptr<LoopQueryKeyframeMessage> queryKeyframe;
    if (queryKeyframeList_.PopBlocking(&queryKeyframe) == false) {
      LOG(INFO) << "Shutting down LoopClosureModule.";
      return;
    }
    loopDetectionTimer.start();

    std::shared_ptr<LoopFrameAndMatches> loopFrame;
    std::shared_ptr<KeyframeInDatabase> queryKeyframeInDatabase;
    loopClosureMethod_->detectLoop(queryKeyframe, queryKeyframeInDatabase,
                                   loopFrame);
    loopDetectionTimer.stop();
    if (loopFrame && outputLoopFrameCallback_) {
      outputLoopFrameCallback_(loopFrame);
    }
    PgoResult pgoResult;
    poseGraphOptTimer.start();
    loopClosureMethod_->addConstraintsAndOptimize(*queryKeyframeInDatabase,
                                                  loopFrame, pgoResult);
    poseGraphOptTimer.stop();
    pgoResults_.Push(pgoResult);
  }
  loopClosureMethod_->saveFinalPgoResults();
}

void LoopClosureModule::publisherLoop() {
  for (;;) {
    // get the result data
    PgoResult result;
    if (pgoResults_.PopBlocking(&result) == false) return;

    // call all user callbacks
    for (auto stateCallback : stateCallbackList_) {
      stateCallback(result.stamp_, result.T_WB_);
    }
  }
}

void LoopClosureModule::shutdown() {
  queryKeyframeList_.Shutdown();
  pgoResults_.Shutdown();
  loopClosureThread_.join();
  publisherThread_.join();
}

}  // namespace okvis
