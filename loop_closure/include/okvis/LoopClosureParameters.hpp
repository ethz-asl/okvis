/**
 * @file LoopClosureParameters.hpp
 * @brief Header file for LoopClosureParameters class which encompasses
 * parameters for loop detection and pose graph optimization.
 */

#ifndef INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_
#define INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_

#include <Eigen/Core>
#include <okvis/class_macros.hpp>
#include <glog/logging.h>

namespace okvis {

/**
 * @brief The PipelineParams base class
 * Sets a common base class for parameters of the pipeline
 * for easy parsing/printing. All parameters in VIO should inherit from
 * this class and implement the print/parseYAML virtual functions.
 */
class PipelineParams {
 public:
  POINTER_TYPEDEFS(PipelineParams);
  explicit PipelineParams(const std::string& name);
  virtual ~PipelineParams() = default;

 public:
  // Parameters of the pipeline must specify how to be parsed.
  virtual bool parseYAML(const std::string& filepath) = 0;

  // Parameters of the pipeline must specify how to be printed.
  virtual void print() const = 0;

 public:
  std::string name_;
};

template <class T>
static void parsePipelineParams(const std::string& params_path,
                                T* pipeline_params) {
  static_assert(std::is_base_of<PipelineParams, T>::value,
                "T must be a class that derives from PipelineParams.");
  // Read/define tracker params.
  if (params_path.empty()) {
    LOG(WARNING) << "No " << pipeline_params->name_
                 << " parameters specified, using default.";
    *pipeline_params = T();  // default params
  } else {
    VLOG(100) << "Using user-specified " << pipeline_params->name_
              << " parameters: " << params_path;
    pipeline_params->parseYAML(params_path);
  }
}

class LoopClosureParameters : public PipelineParams {
 public:
  LoopClosureParameters();
  LoopClosureParameters(const std::string name);
  ~LoopClosureParameters();
  virtual bool parseYAML(const std::string& filepath);
  virtual void print() const;
};
}  // namespace okvis

#endif  // INCLUDE_OKVIS_LOOP_CLOSURE_PARAMETERS_HPP_
