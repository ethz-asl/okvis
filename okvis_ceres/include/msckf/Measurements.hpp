
#ifndef INCLUDE_MSCKF_MEASUREMENTS_HPP_
#define INCLUDE_MSCKF_MEASUREMENTS_HPP_

#include <deque>
#include <vector>
#include <memory>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#pragma GCC diagnostic pop
#pragma GCC diagnostic pop
#include <Eigen/Dense>
#include <okvis/Time.hpp>
#include <okvis/kinematics/Transformation.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
template <typename Scalar>
struct GenericImuMeasurement {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief Default constructor.
  GenericImuMeasurement()
      : timeStamp(0.0),
        gyroscopes(),
        accelerometers() {
  }
  /**
   * @brief Constructor.
   * @param gyroscopes_ Gyroscope measurement.
   * @param accelerometers_ Accelerometer measurement.
   */
  GenericImuMeasurement(Scalar time, Eigen::Matrix<Scalar,3,1> gyroscopes_,
                    Eigen::Matrix<Scalar,3,1> accelerometers_)
      : timeStamp(time),
        gyroscopes(gyroscopes_),
        accelerometers(accelerometers_) {
  }

  GenericImuMeasurement(const GenericImuMeasurement& rhs):
      timeStamp(rhs.timeStamp), gyroscopes(rhs.gyroscopes), accelerometers(rhs.accelerometers)
  {

  }
  GenericImuMeasurement& operator=(const GenericImuMeasurement& rhs)
  {
      timeStamp = rhs.timeStamp;
      gyroscopes = rhs.gyroscopes;
      accelerometers = rhs.accelerometers;
      return *this;
  }

  Scalar timeStamp;      ///< Measurement timestamp
  Eigen::Matrix<Scalar,3,1> gyroscopes;     ///< Gyroscope measurement.
  Eigen::Matrix<Scalar,3,1> accelerometers; ///< Accelerometer measurement.
};

template <typename Scalar>
using GenericImuMeasurementDeque = std::deque<GenericImuMeasurement<Scalar>, Eigen::aligned_allocator<GenericImuMeasurement<Scalar> > >;


}  // namespace okvis

#endif // INCLUDE_MSCKF_MEASUREMENTS_HPP_
