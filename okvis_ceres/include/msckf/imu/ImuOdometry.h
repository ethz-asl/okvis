#ifndef INCLUDE_MSCKF_IMU_ODOMETRY_H_
#define INCLUDE_MSCKF_IMU_ODOMETRY_H_
#include <vector>

#include <okvis/Measurements.hpp>
#include <okvis/Parameters.hpp>
#include <okvis/Time.hpp>
#include <okvis/Variables.hpp>
#include <okvis/assert_macros.hpp>

#include "msckf/imu/odeHybrid.hpp"

namespace okvis {

class IMUOdometry {
  /// \brief The type of the covariance.
  typedef Eigen::Matrix<double, 15, 15> covariance_t;

  /// \brief The type of the information (same matrix dimension as covariance).
  typedef covariance_t information_t;

  /// \brief The type of hte overall Jacobian.
  typedef Eigen::Matrix<double, 15, 15> jacobian_t;

 public:
  /**
   * @brief Propagates pose, speeds and biases with given IMU measurements.
   * @remark This can be used externally to perform propagation
   * @param[in] imuMeasurements All the IMU measurements.
   * @param[in] imuParams The parameters to be used.
   * @param[inout] T_WS Start pose.
   * @param[inout] speedAndBiases Start speed and biases.
   * @param[in] t_start Start time.
   * @param[in] t_end End time.
   * @param[out] covariance Covariance for GIVEN start states.
   * @param[out] jacobian Jacobian w.r.t. start states.
   * @param[in] linearizationPointAtTStart is the first estimates of position
   * p_WS and velocity v_WS at t_start
   * @return Number of integration steps.
   * assume W frame has z axis pointing up
   * Euler approximation is used to incrementally compute the integrals, and
   * the length of integral interval only adversely affect the covariance and
   * jacobian a little.
   */
  static int propagation(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      const IMUErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* covariance_t = 0,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* jacobian = 0,
      const Eigen::Matrix<double, 6, 1>* linearizationPointAtTStart = 0);
  // a copy of the original implementation forreference
  static int propagation_original(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      const IMUErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* covariance_t = 0,
      Eigen::Matrix<double, ceres::ode::OdoErrorStateDim,
                    ceres::ode::OdoErrorStateDim>* jacobian = 0);

  // t_start is greater than t_end
  static int propagationBackward(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS, Eigen::Vector3d& v_WS,
      const IMUErrorModel<double>& iem, const okvis::Time& t_start,
      const okvis::Time& t_end);

  static int propagation_RungeKutta(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBiases& speedAndBias,
      const Eigen::Matrix<double, 27, 1>& vTgTsTa, const okvis::Time& t_start,
      const okvis::Time& t_end,
      Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                    okvis::ceres::ode::OdoErrorStateDim>* P_ptr = 0,
      Eigen::Matrix<double, okvis::ceres::ode::OdoErrorStateDim,
                    okvis::ceres::ode::OdoErrorStateDim>* F_tot_ptr = 0);

  /**
   * @brief propagationBackward_RungeKutta propagate pose, speed and biases.
   * @warning This method assumes that z direction of the world frame is along negative gravity.
   * @param imuMeasurements [t0, t1, ..., t_{n-1}]
   * @param imuParams
   * @param T_WS pose at t_start
   * @param speedAndBias linear velocity and bias at t_start
   * @param vTgTsTa
   * @param t_start
   * @param t_end t_start >= t_end
   * @return number of used IMU measurements
   */
  static int propagationBackward_RungeKutta(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBiases& speedAndBias,
      const Eigen::Matrix<double, 27, 1>& vTgTsTa, const okvis::Time& t_start,
      const okvis::Time& t_end);

  /**
   * @brief interpolateInertialData linearly interpolate inertial readings
   *     at queryTime given imuMeas
   * @param imuMeas has size greater than 0
   * @param iem The intermediate members of iem may be changed.
   * @param queryTime
   * @param queryValue
   * @return false if interpolation is impossible, e.g., in the case of
   *     extrapolation or empty imuMeas
   */
  static bool interpolateInertialData(const okvis::ImuMeasurementDeque& imuMeas,
                                      const IMUErrorModel<double>& iem,
                                      const okvis::Time& queryTime,
                                      okvis::ImuMeasurement& queryValue);

  static int propagation_leutenegger_corrected(
      const okvis::ImuMeasurementDeque& imuMeasurements,
      const okvis::ImuParameters& imuParams,
      okvis::kinematics::Transformation& T_WS,
      okvis::SpeedAndBias& speedAndBiases, const okvis::Time& t_start,
      const okvis::Time& t_end, covariance_t* covariance = 0,
      jacobian_t* jacobian = 0);
}; // class IMUOdometry

/**
 * @brief poseAndVelocityAtObservation for feature i, estimate
 *     $p_B^G(t_{f_i})$, $R_B^G(t_{f_i})$, $v_B^G(t_{f_i})$, and
 *     $\omega_{GB}^B(t_{f_i})$ with imu measurements
 * @param imuMeas cover stateEpoch to the extent of featureTime
 * @param imuAugmentedParams imu params exccept for gyro and accel biases
 * @param imuParameters
 * @param stateEpoch
 * @param featureTime
 * @param T_WB[in/out] in: pose at stateEpoch;
 *     out: pose at stateEpoch + featureTime
 * @param sb[in/out] in: speed and biases at stateEpoch;
 *     out: speed and biases at stateEpoch + featureTime
 * @param interpolatedInertialData[out] inertial measurements at stateEpoch +
 *     featureTime after correction for biases etc.
 */
void poseAndVelocityAtObservation(
    const ImuMeasurementDeque& imuMeas, const double* imuAugmentedParams,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, kinematics::Transformation* T_WB,
    SpeedAndBiases* sb, okvis::ImuMeasurement* interpolatedInertialData,
    bool use_RK4);

/**
 * @brief poseAndLinearVelocityAtObservation Similarly to
 *     poseAndVelocityAtObservation except that the inertial data is not
 *     interpolated and the RK4 propagation is not used.
 * @param imuMeas
 * @param imuAugmentedParams
 * @param imuParameters
 * @param stateEpoch
 * @param featureTime
 * @param T_WB
 * @param sb
 */
void poseAndLinearVelocityAtObservation(
    const ImuMeasurementDeque& imuMeas, const double* imuAugmentedParams,
    const okvis::ImuParameters& imuParameters, const okvis::Time& stateEpoch,
    const okvis::Duration& featureTime, kinematics::Transformation* T_WB,
    SpeedAndBiases* sb);

}  // namespace okvis
#endif // INCLUDE_MSCKF_IMU_ODOMETRY_H_
