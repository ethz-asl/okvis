#include <glog/logging.h>

#include <okvis/LoopClosureModule.hpp>

namespace okvis {
LoopClosureModule::LoopClosureModule()
    : blocking_(false), LoopClosureMethod_(new LoopClosureMethod()) {}

LoopClosureModule::LoopClosureModule(
    std::shared_ptr<okvis::LoopClosureMethod> loopClosureMethod)
    : blocking_(false), LoopClosureMethod_(loopClosureMethod) {}

LoopClosureModule::~LoopClosureModule() {}

void LoopClosureModule::setBlocking(bool blocking) { blocking_ = blocking; }

bool LoopClosureModule::push(
    std::shared_ptr<KeyframeForLoopDetection> queryKeyframe) {
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

void LoopClosureModule::startThreads() {
  loopClosureThread_ = std::thread(&LoopClosureModule::loopClosureLoop, this);
}

void LoopClosureModule::loopClosureLoop() {
  TimerSwitchable loopDetectionTimer("4.1 loopDetection", true);
  TimerSwitchable poseGraphOptTimer("4.2 poseGraphOpt", true);
  for (;;) {
    std::shared_ptr<KeyframeForLoopDetection> queryKeyframe;
    if (queryKeyframeList_.PopBlocking(&queryKeyframe) == false) {
      LOG(INFO) << "Shutting down LoopClosureModule.";
      return;
    }
    if (!queryKeyframe) {
      continue;
    }
    loopDetectionTimer.start();
    std::shared_ptr<LoopFrameAndMatches> outputLoopFrame =
        LoopClosureMethod_->detectLoop(queryKeyframe);
    loopDetectionTimer.stop();
    if (outputLoopFrame && outputLoopFrameCallback_) {
      outputLoopFrameCallback_(outputLoopFrame);
    }
    poseGraphOptTimer.start();
    LoopClosureMethod_->addConstraintsAndOptimize(queryKeyframe,
                                                  outputLoopFrame);
    poseGraphOptTimer.stop();
  }
}

void LoopClosureModule::shutdown() {
  queryKeyframeList_.Shutdown();
  loopClosureThread_.join();
}

}  // namespace okvis
