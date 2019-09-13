#ifndef INCLUDE_MSCKF_MODEL_SWITCH_HPP_
#define INCLUDE_MSCKF_MODEL_SWITCH_HPP_

namespace okvis {
#ifndef MODEL_SWITCH_CASES
#define MODEL_SWITCH_CASES         \
  MODEL_CASES                      \
  default:                         \
    MODEL_DOES_NOT_EXIST_EXCEPTION \
    break;
#endif

#define MODEL_DOES_NOT_EXIST_EXCEPTION \
  throw std::domain_error("Model does not exist");

} // namespace okvis
#endif  // INCLUDE_MSCKF_MODEL_SWITCH_HPP_
