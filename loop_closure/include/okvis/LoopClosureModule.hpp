#ifndef INCLUDE_OKVIS_LOOP_CLOSURE_MODULE_HPP_
#define INCLUDE_OKVIS_LOOP_CLOSURE_MODULE_HPP_

#include <memory>
#include <thread>

#include <okvis/KeyframeForLoopDetection.hpp>
#include <okvis/LoopClosureMethod.hpp>
#include <okvis/threadsafe/ThreadsafeQueue.hpp>
#include <okvis/timing/Timer.hpp>

namespace okvis {
typedef std::function<void(std::shared_ptr<okvis::LoopFrameAndMatches>)>
    OutputLoopFrameCallback;
class LoopClosureModule {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

#ifdef DEACTIVATE_TIMERS
  typedef okvis::timing::DummyTimer TimerSwitchable;
#else
  typedef okvis::timing::Timer TimerSwitchable;
#endif
  LoopClosureModule();

  explicit LoopClosureModule(
      std::shared_ptr<okvis::LoopClosureMethod> loopClosureMethod);

  ~LoopClosureModule();

  void setBlocking(bool blocking);

  bool push(std::shared_ptr<KeyframeForLoopDetection> queryKeyframe);

  void setOutputLoopFrameCallback(
      const OutputLoopFrameCallback& outputCallback);

  void startThreads();

  void loopClosureLoop();

  void shutdown();

 private:
  bool blocking_;  ///< Blocking option. Whether the addMeasurement() functions
                   ///< should wait until proccessing is complete.

  okvis::threadsafe::ThreadSafeQueue<
      std::shared_ptr<okvis::KeyframeForLoopDetection>>
      queryKeyframeList_;

  OutputLoopFrameCallback
      outputLoopFrameCallback_;  ///< output loop frame and matches callback
                                 ///< function.

  std::shared_ptr<LoopClosureMethod> LoopClosureMethod_;

  std::thread loopClosureThread_;

  const size_t maxQueryKeyframeQueueSize_ = 5;
};
}  // namespace okvis
#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_MODULE_HPP_
