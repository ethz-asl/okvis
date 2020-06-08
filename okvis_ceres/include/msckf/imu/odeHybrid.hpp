/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Jan 7, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file ode.hpp
 * @brief File for ODE integration functionality.
 * @author Stefan Leutenegger
 */

#ifndef INCLUDE_MSCKF_ODE_HYBRID_HPP_
#define INCLUDE_MSCKF_ODE_HYBRID_HPP_

#include <msckf/imu/ImuErrorModel.h>

#include <Eigen/Core>

#include <okvis/FrameTypedefs.hpp>
#include <okvis/Measurements.hpp>
#include <okvis/Variables.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ode/ode.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {
/// \brief ode Namespace for functionality related to ODE integration
/// implemented in okvis.
namespace ode {

const int kNavErrorStateDim = 9;
// Note this function assume that the W frame is with z up, negative gravity
// direction, because in computing sb_dot and G world-centric velocities
__inline__ void evaluateContinuousTimeOde(
    const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double g,
    const Eigen::Vector3d& p_WS_W, const Eigen::Quaterniond& q_WS,
    const okvis::SpeedAndBiases& sb,
    const ImuErrorModel<double>& iem, Eigen::Vector3d& p_WS_W_dot,
    Eigen::Vector4d& q_WS_dot, okvis::SpeedAndBiases& sb_dot,
    Eigen::MatrixXd* F_c_ptr = 0) {
  Eigen::Vector3d omega_S;
  Eigen::Vector3d acc_S;
  iem.estimate(gyr, acc, &omega_S, &acc_S);

  // nonlinear states
  // start with the pose
  p_WS_W_dot = sb.head<3>();

  // now the quaternion
  Eigen::Vector4d dq;
  q_WS_dot.head<3>() = 0.5 * omega_S;
  q_WS_dot[3] = 0.0;
  Eigen::Matrix3d C_WS = q_WS.toRotationMatrix();

  // the rest is straightforward
  // consider Earth's radius. Model the Earth as a sphere, since we neither
  // know the position nor yaw (except if coupled with GPS and magnetometer).
  Eigen::Vector3d G =
      -p_WS_W - Eigen::Vector3d(0, 0, 6371009);  // vector to Earth center
  sb_dot.head<3>() = (C_WS * acc_S + g * G.normalized());  // s
  // biases
  sb_dot.tail<6>().setZero();

  // linearized system:
  if (F_c_ptr) {
    F_c_ptr->setZero();
    F_c_ptr->block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
    //    F_c_ptr->block<3, 3>(3, 9) -= C_WS;
    F_c_ptr->block<3, 3>(6, 3) -= okvis::kinematics::crossMx(C_WS * acc_S);
    //    F_c_ptr->block<3, 3>(6, 12) -= C_WS;
    Eigen::Matrix<double, 9, 6> N = Eigen::Matrix<double, 9, 6>::Zero();
    N.block<3, 3>(3, 0) = C_WS;
    N.block<3, 3>(6, 3) = C_WS;
    int colsF = F_c_ptr->cols();
    Eigen::Matrix<double, 6, 6 + 27> dwaB_dbgbaSTS;
    iem.dwa_B_dbgbaSTS(omega_S, acc_S, dwaB_dbgbaSTS);
    F_c_ptr->block(0, 9, 9, colsF - 9) = N * dwaB_dbgbaSTS.rightCols(colsF - 9);
  }
}

// p_WS_W, q_WS, sb are states at k
/* * covariance error states $\Delta y = \Delta[\mathbf{p}_{WS}^W,
\Delta\mathbf{\alpha}, \Delta\mathbf{v}_S^W,
 * \Delta\mathbf{b_g}, \Delta\mathbf{b_a}, \Delta\mathbf{Tg}, \Delta\mathbf{Ts},
\Delta\mathbf{Ta}]$
 * $\tilde{R}_s^w=(I-[\delta\alpha^w]_\times)R_s^w$
 * this DEFINITION OF ROTATION ERROR is the same in Mingyang Li's later
publications, okvis, and huai ION GNSS+ 2015 $y = [\mathbf{p}_{WS}^W,
\mathbf{q}_S^W, \mathbf{v}_S^W, \mathbf{b_g}, \mathbf{b_a}]$ $u =
[\mathbf{\omega}_{WS}^S,\mathbf{a}^S]$ $h = t_{n+1}-t_n$
$\mathbf{p}_{WS}^W \oplus = \mathbf{p}_{WS}^W +$
$\mathbf{v}_{WS}^W \oplus = \mathbf{v}_{WS}^W +$
$\mathbf{q}_{S}^W \oplus \mathbf{\omega}_{WS}^S h/2 =
\mathbf{q}_{S}^W\begin{bmatrix}
cos(\mathbf{\omega}_{WS}^S h/2) \\
sin(\mathbf{\omega}_{WS}^S
h/2)\frac{\mathbf{\omega}_{WS}^S/2}{\vert\mathbf{\omega}_{WS}^S/2\vert}
\end{bmatrix}$

$k_1 = f(t_n,y_n,u_n)$
$k_2 = f(t_n+h/2,y_n\oplus k_1 h/2 ,(u_n +u_{n+1})/2)$
$k_3 = f(t_n+h/2,y_n\oplus k_2 h/2 ,(u_n +u_{n+1})/2)$
$k_4 = f(t_n+h,y_n\oplus k_3 h , u_{n+1})$
$y_{n+1}=y_n\oplus\left(h(k_1 +2k_2 +2k_3 +k_4)/6 \right )$
Caution: provide both F_tot_ptr(e.g., identity) and P_ptr(e.g., zero matrix) if
covariance is to be computed
*/
__inline__ void integrateOneStep_RungeKutta(
    const Eigen::Vector3d& gyr_0, const Eigen::Vector3d& acc_0,
    const Eigen::Vector3d& gyr_1, const Eigen::Vector3d& acc_1, double g,
    double sigma_g_c, double sigma_a_c, double sigma_gw_c, double sigma_aw_c,
    double dt, Eigen::Vector3d& p_WS_W, Eigen::Quaterniond& q_WS,
    okvis::SpeedAndBiases& sb, const ImuErrorModel<double>& iem,
    Eigen::MatrixXd* P_ptr = 0,
    Eigen::MatrixXd* F_tot_ptr = 0) {
  Eigen::Vector3d k1_p_WS_W_dot;
  Eigen::Vector4d k1_q_WS_dot;

  okvis::SpeedAndBiases k1_sb_dot;
  int covRows = 0;
  Eigen::MatrixXd k1_F_c;
  Eigen::MatrixXd k2_F_c;
  Eigen::MatrixXd k3_F_c;
  Eigen::MatrixXd k4_F_c;
  Eigen::MatrixXd* k1_F_c_ptr = nullptr;
  Eigen::MatrixXd* k2_F_c_ptr = nullptr;
  Eigen::MatrixXd* k3_F_c_ptr = nullptr;
  Eigen::MatrixXd* k4_F_c_ptr = nullptr;
  if (P_ptr || F_tot_ptr) {
      covRows = P_ptr ? P_ptr->rows() : F_tot_ptr->rows();
      k1_F_c.resize(covRows, covRows);
      k2_F_c.resize(covRows, covRows);
      k3_F_c.resize(covRows, covRows);
      k4_F_c.resize(covRows, covRows);
      k1_F_c_ptr = &k1_F_c;
      k2_F_c_ptr = &k2_F_c;
      k3_F_c_ptr = &k3_F_c;
      k4_F_c_ptr = &k4_F_c;
  }

  evaluateContinuousTimeOde(gyr_0, acc_0, g, p_WS_W, q_WS, sb, iem,
                            k1_p_WS_W_dot, k1_q_WS_dot, k1_sb_dot, k1_F_c_ptr);

  Eigen::Vector3d p_WS_W1 = p_WS_W;
  Eigen::Quaterniond q_WS1 = q_WS;
  okvis::SpeedAndBiases sb1 = sb;
  // state propagation:
  p_WS_W1 += k1_p_WS_W_dot * 0.5 * dt;
  Eigen::Quaterniond dq;
  double theta_half = k1_q_WS_dot.head<3>().norm() * 0.5 * dt;
  double sinc_theta_half = sinc(theta_half);
  double cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * k1_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  q_WS1 = q_WS * dq;
  sb1 += k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k2_p_WS_W_dot;
  Eigen::Vector4d k2_q_WS_dot;
  okvis::SpeedAndBiases k2_sb_dot;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W1, q_WS1, sb1, iem, k2_p_WS_W_dot,
                            k2_q_WS_dot, k2_sb_dot, k2_F_c_ptr);

  Eigen::Vector3d p_WS_W2 = p_WS_W;
  Eigen::Quaterniond q_WS2 = q_WS;
  okvis::SpeedAndBiases sb2 = sb;
  // state propagation:
  p_WS_W2 += k2_p_WS_W_dot * 0.5 * dt;
  theta_half = k2_q_WS_dot.head<3>().norm() * 0.5 * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * k2_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS2 = q_WS2 * dq;
  sb2 += k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k3_p_WS_W_dot;
  Eigen::Vector4d k3_q_WS_dot;
  okvis::SpeedAndBiases k3_sb_dot;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W2, q_WS2, sb2, iem, k3_p_WS_W_dot,
                            k3_q_WS_dot, k3_sb_dot, k3_F_c_ptr);

  Eigen::Vector3d p_WS_W3 = p_WS_W;
  Eigen::Quaterniond q_WS3 = q_WS;
  okvis::SpeedAndBiases sb3 = sb;
  // state propagation:
  p_WS_W3 += k3_p_WS_W_dot * dt;
  theta_half = k3_q_WS_dot.head<3>().norm() * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * k3_q_WS_dot.head<3>() * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS3 = q_WS3 * dq;
  sb3 += k3_sb_dot * dt;

  Eigen::Vector3d k4_p_WS_W_dot;
  Eigen::Vector4d k4_q_WS_dot;
  okvis::SpeedAndBiases k4_sb_dot;
  evaluateContinuousTimeOde(gyr_1, acc_1, g, p_WS_W3, q_WS3, sb3, iem,
                            k4_p_WS_W_dot, k4_q_WS_dot, k4_sb_dot, k4_F_c_ptr);

  // now assemble
  p_WS_W +=
      (k1_p_WS_W_dot + 2 * (k2_p_WS_W_dot + k3_p_WS_W_dot) + k4_p_WS_W_dot) *
      dt / 6.0;
  Eigen::Vector3d theta_half_vec =
      (k1_q_WS_dot.head<3>() +
       2 * (k2_q_WS_dot.head<3>() + k3_q_WS_dot.head<3>()) +
       k4_q_WS_dot.head<3>()) *
      dt / 6.0;
  theta_half = theta_half_vec.norm();
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * theta_half_vec;
  dq.w() = cos_theta_half;
  q_WS = q_WS * dq;
  sb += (k1_sb_dot + 2 * (k2_sb_dot + k3_sb_dot) + k4_sb_dot) * dt / 6.0;

  q_WS.normalize();  // do not accumulate errors!

  if (F_tot_ptr) {
    // compute state transition matrix, note $\frac{d\Phi(t, t_0)}{dt}=
    // F(t)\Phi(t, t_0)$
    Eigen::MatrixXd& F_tot = *F_tot_ptr;
    const Eigen::MatrixXd& J1 = k1_F_c;
    const Eigen::MatrixXd J2 =
        k2_F_c * (Eigen::MatrixXd::Identity(covRows, covRows) + 0.5 * dt * J1);
    const Eigen::MatrixXd J3 =
        k3_F_c * (Eigen::MatrixXd::Identity(covRows, covRows) + 0.5 * dt * J2);
    const Eigen::MatrixXd J4 =
        k4_F_c * (Eigen::MatrixXd::Identity(covRows, covRows) + dt * J3);
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(covRows, covRows) +
                        dt * (J1 + 2 * (J2 + J3) + J4) / 6.0;
    F_tot =
        (F * F_tot)
            .eval();  // F is $\Phi(t_k, t_{k-1})$, F_tot is $\Phi(t_k, t_{0})$

    if (P_ptr) {
      Eigen::MatrixXd& cov = *P_ptr;
      cov = F * (cov * F.transpose()).eval();

      // add process noise
      const double Q_g = sigma_g_c * sigma_g_c * dt;
      const double Q_a = sigma_a_c * sigma_a_c * dt;
      const double Q_gw = sigma_gw_c * sigma_gw_c * dt;
      const double Q_aw = sigma_aw_c * sigma_aw_c * dt;
      cov(3, 3) += Q_g;
      cov(4, 4) += Q_g;
      cov(5, 5) += Q_g;
      cov(6, 6) += Q_a;
      cov(7, 7) += Q_a;
      cov(8, 8) += Q_a;
      cov(9, 9) += Q_gw;
      cov(10, 10) += Q_gw;
      cov(11, 11) += Q_gw;
      cov(12, 12) += Q_aw;
      cov(13, 13) += Q_aw;
      cov(14, 14) += Q_aw;

      // force symmetric
      // huai: this may help keep cov positive semi-definite after propagation
      cov = 0.5 * cov + 0.5 * cov.transpose().eval();
    }
  }
}

/* p_WS_W, q_WS, sb are states at k+1, dt= t(k+1) -t(k)
$y = [\mathbf{p}_{WS}^W, \mathbf{q}_S^W, \mathbf{v}_S^W, \mathbf{b_g},
\mathbf{b_a}]$ $u = [\mathbf{\omega}_{WS}^S,\mathbf{a}^S]$ $h = t_{n}-t_{n+1}$
$\mathbf{p}_{WS}^W \oplus = \mathbf{p}_{WS}^W +$
$\mathbf{v}_{WS}^W \oplus = \mathbf{v}_{WS}^W +$
$\mathbf{q}_{S}^W \oplus \mathbf{\omega}_{WS}^S h/2 =
\mathbf{q}_{S}^W\begin{bmatrix}
cos(\mathbf{\omega}_{WS}^S h/2) \\
sin(\mathbf{\omega}_{WS}^S
h/2)\frac{\mathbf{\omega}_{WS}^S/2}{\vert\mathbf{\omega}_{WS}^S/2\vert}
\end{bmatrix}$

$k_1 = f(t_{n+1},y_{n+1},u_{n+1})$
$k_2 = f(t_{n+1}+h/2,y_{n+1}\oplus k_1 h/2 ,(u_n +u_{n+1})/2)$
$k_3 = f(t_{n+1}+h/2,y_{n+1}\oplus k_2 h/2 ,(u_n +u_{n+1})/2)$
$k_4 = f(t_n,y_{n+1}\oplus k_3 h , u_{n})$
$y_{n}=y_{n+1}\oplus\left(h(k_1 +2k_2 +2k_3 +k_4)/6 \right )$
*/
__inline__ void integrateOneStepBackward_RungeKutta(
    const Eigen::Vector3d& gyr_0, const Eigen::Vector3d& acc_0,
    const Eigen::Vector3d& gyr_1, const Eigen::Vector3d& acc_1, double g,
    double sigma_g_c, double sigma_a_c, double sigma_gw_c, double sigma_aw_c,
    double dt, Eigen::Vector3d& p_WS_W, Eigen::Quaterniond& q_WS,
    okvis::SpeedAndBiases& sb, const ImuErrorModel<double>& iem,
    Eigen::MatrixXd* P_ptr = 0,
    Eigen::MatrixXd* F_tot_ptr = 0) {
  Eigen::Vector3d k1_p_WS_W_dot;
  Eigen::Vector4d k1_q_WS_dot;
  okvis::SpeedAndBiases k1_sb_dot;

  int covRows = 0;
  Eigen::MatrixXd k1_F_c;
  Eigen::MatrixXd k2_F_c;
  Eigen::MatrixXd k3_F_c;
  Eigen::MatrixXd k4_F_c;
  Eigen::MatrixXd* k1_F_c_ptr = nullptr;
  Eigen::MatrixXd* k2_F_c_ptr = nullptr;
  Eigen::MatrixXd* k3_F_c_ptr = nullptr;
  Eigen::MatrixXd* k4_F_c_ptr = nullptr;
  if (P_ptr || F_tot_ptr) {
      covRows = P_ptr ? P_ptr->rows() : F_tot_ptr->rows();
      k1_F_c.resize(covRows, covRows);
      k2_F_c.resize(covRows, covRows);
      k3_F_c.resize(covRows, covRows);
      k4_F_c.resize(covRows, covRows);
      k1_F_c_ptr = &k1_F_c;
      k2_F_c_ptr = &k2_F_c;
      k3_F_c_ptr = &k3_F_c;
      k4_F_c_ptr = &k4_F_c;
  }

  evaluateContinuousTimeOde(gyr_1, acc_1, g, p_WS_W, q_WS, sb, iem,
                            k1_p_WS_W_dot, k1_q_WS_dot, k1_sb_dot, k1_F_c_ptr);

  Eigen::Vector3d p_WS_W1 = p_WS_W;
  Eigen::Quaterniond q_WS1 = q_WS;
  okvis::SpeedAndBiases sb1 = sb;
  // state propagation:
  p_WS_W1 -= k1_p_WS_W_dot * 0.5 * dt;
  Eigen::Quaterniond dq;
  double theta_half = -k1_q_WS_dot.head<3>().norm() * 0.5 * dt;
  double sinc_theta_half = sinc(theta_half);
  double cos_theta_half = cos(theta_half);
  dq.vec() = -sinc_theta_half * k1_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  q_WS1 = q_WS * dq;
  sb1 -= k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k2_p_WS_W_dot;
  Eigen::Vector4d k2_q_WS_dot;
  okvis::SpeedAndBiases k2_sb_dot;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W1, q_WS1, sb1, iem, k2_p_WS_W_dot,
                            k2_q_WS_dot, k2_sb_dot, k2_F_c_ptr);

  Eigen::Vector3d p_WS_W2 = p_WS_W;
  Eigen::Quaterniond q_WS2 = q_WS;
  okvis::SpeedAndBiases sb2 = sb;
  // state propagation:
  p_WS_W2 -= k2_p_WS_W_dot * 0.5 * dt;
  theta_half = -k2_q_WS_dot.head<3>().norm() * 0.5 * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = -sinc_theta_half * k2_q_WS_dot.head<3>() * 0.5 * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS2 = q_WS2 * dq;
  sb2 -= k1_sb_dot * 0.5 * dt;

  Eigen::Vector3d k3_p_WS_W_dot;
  Eigen::Vector4d k3_q_WS_dot;
  okvis::SpeedAndBiases k3_sb_dot;
  evaluateContinuousTimeOde(0.5 * (gyr_0 + gyr_1), 0.5 * (acc_0 + acc_1), g,
                            p_WS_W2, q_WS2, sb2, iem, k3_p_WS_W_dot,
                            k3_q_WS_dot, k3_sb_dot, k3_F_c_ptr);

  Eigen::Vector3d p_WS_W3 = p_WS_W;
  Eigen::Quaterniond q_WS3 = q_WS;
  okvis::SpeedAndBiases sb3 = sb;
  // state propagation:
  p_WS_W3 -= k3_p_WS_W_dot * dt;
  theta_half = -k3_q_WS_dot.head<3>().norm() * dt;
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = -sinc_theta_half * k3_q_WS_dot.head<3>() * dt;
  dq.w() = cos_theta_half;
  // std::cout<<dq.transpose()<<std::endl;
  q_WS3 = q_WS3 * dq;
  sb3 -= k3_sb_dot * dt;

  Eigen::Vector3d k4_p_WS_W_dot;
  Eigen::Vector4d k4_q_WS_dot;
  okvis::SpeedAndBiases k4_sb_dot;
  evaluateContinuousTimeOde(gyr_0, acc_0, g, p_WS_W3, q_WS3, sb3, iem,
                            k4_p_WS_W_dot, k4_q_WS_dot, k4_sb_dot, k4_F_c_ptr);

  // now assemble
  p_WS_W -=
      (k1_p_WS_W_dot + 2 * (k2_p_WS_W_dot + k3_p_WS_W_dot) + k4_p_WS_W_dot) *
      dt / 6.0;
  Eigen::Vector3d theta_half_vec =
      -(k1_q_WS_dot.head<3>() +
        2 * (k2_q_WS_dot.head<3>() + k3_q_WS_dot.head<3>()) +
        k4_q_WS_dot.head<3>()) *
      dt / 6.0;
  theta_half = theta_half_vec.norm();
  sinc_theta_half = sinc(theta_half);
  cos_theta_half = cos(theta_half);
  dq.vec() = sinc_theta_half * theta_half_vec;
  dq.w() = cos_theta_half;
  q_WS = q_WS * dq;
  sb -= (k1_sb_dot + 2 * (k2_sb_dot + k3_sb_dot) + k4_sb_dot) * dt / 6.0;

  q_WS.normalize();  // do not accumulate errors!

  if (F_tot_ptr) {
    assert(false);  // the following section is not well perused and tested
    // compute state transition matrix, note $\frac{d\Phi(t, t_0)}{dt}=
    // F(t)\Phi(t, t_0)$
    Eigen::MatrixXd& F_tot = *F_tot_ptr;
    const Eigen::MatrixXd& J1 = k1_F_c;
    const Eigen::MatrixXd J2 =
        k2_F_c * (Eigen::MatrixXd::Identity(covRows, covRows) - 0.5 * dt * J1);
    const Eigen::MatrixXd J3 =
        k3_F_c * (Eigen::MatrixXd::Identity(covRows, covRows) - 0.5 * dt * J2);
    const Eigen::MatrixXd J4 =
        k4_F_c * (Eigen::MatrixXd::Identity(covRows, covRows) - dt * J3);
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(covRows, covRows) -
                        dt * (J1 + 2 * (J2 + J3) + J4) / 6.0;
    F_tot = (F * F_tot).eval();

    if (P_ptr) {
      Eigen::MatrixXd& cov = *P_ptr;
      cov = F * (cov * F.transpose()).eval();

      // add process noise
      const double Q_g = sigma_g_c * sigma_g_c * dt;
      const double Q_a = sigma_a_c * sigma_a_c * dt;
      const double Q_gw = sigma_gw_c * sigma_gw_c * dt;
      const double Q_aw = sigma_aw_c * sigma_aw_c * dt;
      cov(3, 3) += Q_g;
      cov(4, 4) += Q_g;
      cov(5, 5) += Q_g;
      cov(6, 6) += Q_a;
      cov(7, 7) += Q_a;
      cov(8, 8) += Q_a;
      cov(9, 9) += Q_gw;
      cov(10, 10) += Q_gw;
      cov(11, 11) += Q_gw;
      cov(12, 12) += Q_aw;
      cov(13, 13) += Q_aw;
      cov(14, 14) += Q_aw;

      // force symmetric - TODO: is this really needed here?
      // cov = 0.5 * cov + 0.5 * cov.transpose().eval();
    }
  }
}
}  // namespace ode

}  // namespace ceres
}  // namespace okvis

#endif // INCLUDE_MSCKF_ODE_HYBRID_HPP_
