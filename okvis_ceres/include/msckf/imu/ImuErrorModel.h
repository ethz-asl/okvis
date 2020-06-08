#ifndef IMU_ERROR_MODEL_H_
#define IMU_ERROR_MODEL_H_

#include <Eigen/Dense>
// accelerometer and gyro error model by drawing inspirations from Mingyang Li
// ICRA 2014, Titterton and Weston 12.5.2, and Tedaldi ICRA 2014 A robust and
// easy to implement method. Here we follow exactly the model used in
// Mingyang Li ICRA 2014 and Shelley 2014 master thesis
template <class Scalar>
class ImuErrorModel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // a volatile class to estimate linear acceleration and angular rate given
  // measurements, or predict measurements given estimated values
  Eigen::Matrix<Scalar, 3, 1> b_g;
  Eigen::Matrix<Scalar, 3, 1> b_a;

  Eigen::Matrix<Scalar, 3, 3> S_g;  // S_g + I_3 = T_g
  Eigen::Matrix<Scalar, 3, 3> T_s;  // T_s
  Eigen::Matrix<Scalar, 3, 3> S_a;  // S_a + I_3 = T_a

  Eigen::Matrix<Scalar, 3, 3> invT_g;  // inverse of T_g
  Eigen::Matrix<Scalar, 3, 3> invT_a;  // inverse of T_a

  // intermediate variables
//  Eigen::Matrix<Scalar, 3, 1> a_est;
//  Eigen::Matrix<Scalar, 3, 1> w_est;
//  Eigen::Matrix<Scalar, 3, 1> a_obs;
//  Eigen::Matrix<Scalar, 3, 1> w_obs;

  ImuErrorModel(const Eigen::Matrix<Scalar, 6, 1>& b_ga =
                    Eigen::Matrix<Scalar, 6, 1>::Zero());

  ImuErrorModel(const Eigen::Matrix<Scalar, 6, 1>& b_ga,
                const Eigen::Matrix<Scalar, -1, 1>& shapeMatrices,
                bool bTgTsTa = true);

//  ImuErrorModel(const Eigen::Matrix<Scalar, 27, 1>& vSaSgTs,
//                const Eigen::Matrix<Scalar, 6, 1>& b_ag);

  // copy constructor
  ImuErrorModel(const ImuErrorModel<Scalar>& iem);

  ImuErrorModel<Scalar>& operator=(const ImuErrorModel<Scalar>& rhs);

  void setBg(const Eigen::Matrix<Scalar, 3, 1>& bg);

  void setBa(const Eigen::Matrix<Scalar, 3, 1>& ba);

//  void estimate(const Eigen::Matrix<Scalar, 3, 1>& w_m,
//                const Eigen::Matrix<Scalar, 3, 1>& a_m);

  void estimate(const Eigen::Matrix<Scalar, 3, 1>& w_m,
                const Eigen::Matrix<Scalar, 3, 1>& a_m,
                Eigen::Matrix<Scalar, 3, 1>* w_est,
                Eigen::Matrix<Scalar, 3, 1>* a_est) const;

  void predict(const Eigen::Matrix<Scalar, 3, 1>& w_s,
               const Eigen::Matrix<Scalar, 3, 1>& a_s,
               Eigen::Matrix<Scalar, 3, 1>* w_m,
               Eigen::Matrix<Scalar, 3, 1>* a_m) const;

  // the following functions refer to Michael Andrew Shelley master thesis 2014
  // with some corrections calculate $\frac{\partial{T_{3\times3}}}{\partial
  // \vec{T}_9}\vec{a}_{3}$
  Eigen::Matrix<Scalar, 3, 9> dmatrix3_dvector9_multiply(
      const Eigen::Matrix<Scalar, 3, 1> rhs) const;

  // calculate $\frac{\partial\ \omega_{WB}^B}{\partial {(b_g, b_a)}}$
  Eigen::Matrix<Scalar, 3, 6> domega_B_dbgba() const;
  // calculate $\frac{\partial\ \omega_{WB}^B}{\partial {(\vec{T}_g, \vec{T}_s,
  // \vec{T}_a)}}$ which is also
  //$\frac{\partial\ \omega_{WB}^B}{\partial {(\vec{S}_g, \vec{T}_s,
  //\vec{S}_a)}}$
  // Note call this function after estimate because it requires the latest
  // a_est, and w_est
  Eigen::Matrix<Scalar, 3, 27> domega_B_dSgTsSa(
      const Eigen::Matrix<Scalar, 3, 1>& w_est,
      const Eigen::Matrix<Scalar, 3, 1>& a_est) const;

  // calculate $\frac{\partial\ a^B}{\partial {(b_g, b_a)}}$
  Eigen::Matrix<Scalar, 3, 6> dacc_B_dbgba() const;
  // calculate $\frac{\partial\ a^B}{\partial {(\vec{T}_g, \vec{T}_s,
  // \vec{T}_a)}}$ which is also
  //$\frac{\partial\ a^B}{\partial {(\vec{S}_g, \vec{T}_s, \vec{S}_a)}}$
  // Note call this function after estimate because it requires the latest
  // a_est, and w_est
  Eigen::Matrix<Scalar, 3, 27> dacc_B_dSgTsSa(
      const Eigen::Matrix<Scalar, 3, 1>& a_est) const;

  // calculate $\frac{\partial\ [\omega_{WB}^B, a^B]}{\partial {(b_g, b_a,
  // \vec{T}_g, \vec{T}_s, \vec{T}_a)}}$ Note call this function after estimate
  // because it requires the latest a_est, and w_est
  void dwa_B_dbgbaSTS(const Eigen::Matrix<Scalar, 3, 1>& w_est,
                      const Eigen::Matrix<Scalar, 3, 1>& a_est,
                      Eigen::Matrix<Scalar, 6, 27 + 6>& output) const;
};

#include "../implementation/ImuErrorModel.h"
#endif
