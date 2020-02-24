#ifndef OKVIS_INCLUDE_FULL_STATE_WITH_EXTRINSICS_AS_CALLBACK_HPP_
#define OKVIS_INCLUDE_FULL_STATE_WITH_EXTRINSICS_AS_CALLBACK_HPP_

#include <fstream>
#include <iomanip>
#include <string>

#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

namespace okvis {
class FullStateWithExtrinsicsAsCallback {
 public:
  void save(
      const okvis::Time &t, const okvis::kinematics::Transformation &T_WS,
      const Eigen::Matrix<double, 9, 1> &speedAndBiases,
      const Eigen::Matrix<double, 3, 1> & /*omega_S*/,
      const int frameIdInSource,
      const std::vector<
          okvis::kinematics::Transformation,
          Eigen::aligned_allocator<okvis::kinematics::Transformation>>
          &extrinsics);

  FullStateWithExtrinsicsAsCallback(const std::string &output_file);

  ~FullStateWithExtrinsicsAsCallback();
private:
  std::string output_file_;
  std::ofstream output_stream_;
};
}
#endif // OKVIS_INCLUDE_FULL_STATE_WITH_EXTRINSICS_AS_CALLBACK_HPP_
