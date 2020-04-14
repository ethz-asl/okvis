#include <okvis/LoopClosureParameters.hpp>
#include <okvis/LoopClosureMethod.hpp>

namespace okvis {
PipelineParams::PipelineParams(const std::string& name) : name_(name) {}

LoopClosureParameters::LoopClosureParameters() :
  PipelineParams("Base Loop Closure Parameters") {}

LoopClosureParameters::LoopClosureParameters(const std::string name) :
  PipelineParams(name) {}

LoopClosureParameters::~LoopClosureParameters() {}

bool LoopClosureParameters::parseYAML(const std::string& /*filepath*/) {
  return true;
}

void LoopClosureParameters::print() const {

}
}  // namespace okvis
