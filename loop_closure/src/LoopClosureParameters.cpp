#include <okvis/LoopClosureParameters.hpp>
#include <okvis/LoopClosureMethod.hpp>

namespace okvis {
LoopClosureParameters::LoopClosureParameters() :
    methodId(okvis::LoopClosureMethod::kMethodId) {}

LoopClosureParameters::~LoopClosureParameters() {}
}  // namespace okvis
