#include "msckf/InitialNavState.hpp"

namespace okvis {

InitialNavState::InitialNavState()
    : initWithExternalSource_(false),
      p_WS(0, 0, 0),
      q_WS(1, 0, 0, 0),
      v_WS(0, 0, 0),
      std_p_WS(0.01, 0.01, 0.01),
      std_q_WS(M_PI / 180, M_PI / 180, 3 * M_PI / 180),
      std_v_WS(0.1, 0.1, 0.1) {}

// v_WS, and std_v_WS are to be recalculated later according to updated p_WS and
// q_ws
InitialNavState::InitialNavState(const okvis::InitialState& rhs)
    : initWithExternalSource_(rhs.bUseExternalInitState),
      stateTime(rhs.stateTime),
      p_WS(0, 0, 0),
      q_WS(rhs.q_WS),
      v_WS(rhs.v_WS),
      std_p_WS(rhs.std_p_WS),
      std_q_WS(rhs.std_q_WS),
      std_v_WS(rhs.std_v_WS) {}

void InitialNavState::updatePose(const okvis::kinematics::Transformation& T_WS,
                                 const okvis::Time state_time) {
  stateTime = state_time;
  p_WS = T_WS.r();
  q_WS = T_WS.q();
}

InitialNavState& InitialNavState::operator=(const InitialNavState& other) {
  if (&other == this) return *this;
  initWithExternalSource_ = other.initWithExternalSource_;
  stateTime = other.stateTime;
  p_WS = other.p_WS;
  q_WS = other.q_WS;
  v_WS = other.v_WS;
  std_p_WS = other.std_p_WS;
  std_q_WS = other.std_q_WS;
  std_v_WS = other.std_v_WS;
  return *this;
}

}  // namespace okvis
