#ifndef INCLUDE_MSCKF_BOUNDED_IMU_DEQUE_HPP_
#define INCLUDE_MSCKF_BOUNDED_IMU_DEQUE_HPP_

#include <mutex>
#include <okvis/Measurements.hpp>

namespace okvis {
// keep measurements streamed from one IMU
class BoundedImuDeque {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  BoundedImuDeque();
  ~BoundedImuDeque();
  // append the imu_segment meas to the end of imu_meas_
  // assume the imu segment meas have strictly increasing timestamps
  // though the timestamps in imu_segment and imu_meas_ may be duplicate
  int push_back(const okvis::ImuMeasurementDeque& imu_segment);
  int pop_front(const okvis::Time& eraseUtil);
  const okvis::ImuMeasurementDeque find(const okvis::Time& begin_time,
                                        const okvis::Time& end_time) const;

  const okvis::ImuMeasurementDeque findWindow(
      const okvis::Time& center_time, const okvis::Duration& half_window) const;
  const okvis::ImuMeasurementDeque& getAllImuMeasurements() const;

 private:
  okvis::ImuMeasurementDeque imu_meas_;
};

/**
 * @brief getImuMeasurements Get a subset of IMU readings falling into
 *     BeginTime and EndTime. If possible, extend the IMU readings
 *     by one on each side.
 * @param imuDataBeginTime
 * @param imuDataEndTime
 * @param imuMeasurements_
 * @param imuMeasurements_mutex_
 * @return empty if no IMU readings fall into BeginTime and EndTime.
 */
okvis::ImuMeasurementDeque getImuMeasurements(
    const okvis::Time& imuDataBeginTime, const okvis::Time& imuDataEndTime,
    const okvis::ImuMeasurementDeque& imuMeasurements_,
    std::mutex* imuMeasurements_mutex_ = nullptr);

// Remove IMU measurements from the internal buffer.
int deleteImuMeasurements(const okvis::Time& eraseUntil,
                          okvis::ImuMeasurementDeque& imuMeasurements_,
                          std::mutex* imuMeasurements_mutex_ = nullptr);
}  // namespace okvis
#endif  // INCLUDE_MSCKF_BOUNDED_IMU_DEQUE_HPP_
