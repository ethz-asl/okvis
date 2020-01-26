#ifndef INCLUDE_MSCKF_EXTRINSIC_MODELS_HPP_
#define INCLUDE_MSCKF_EXTRINSIC_MODELS_HPP_

#include <Eigen/Core>
#include <msckf/JacobianHelpers.hpp>
#include <msckf/ModelSwitch.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

namespace okvis {
// TODO(jhuai): remove ExtrinsicFixed
class ExtrinsicFixed {
 public:
  static const int kModelId = 0;
  static const size_t kNumParams = 0;
  static const size_t kGlobalDim = 0;
  static inline int getMinimalDim() { return kNumParams; }
  static inline Eigen::MatrixXd initCov(double /*sigma_translation*/,
                                        double /*sigma_orientation*/) {
    return Eigen::MatrixXd();
  }
  static void dpC_dExtrinsic(const Eigen::Vector3d& /*pC*/,
                             const Eigen::Matrix3d& /*R_CB*/,
                             Eigen::MatrixXd* dpC_dT,
                             const Eigen::Matrix3d* /*R_CfCa*/,
                             const Eigen::Vector4d* /*ab1rho*/) {
    *dpC_dT = Eigen::MatrixXd();
  }
  static void updateState(const Eigen::Vector3d& r, const Eigen::Quaterniond& q,
                          const Eigen::VectorXd& /*delta*/,
                          Eigen::Vector3d* r_delta,
                          Eigen::Quaterniond* q_delta) {
    *r_delta = r;
    *q_delta = q;
  }

  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d& /*pC*/,
                             const Eigen::Matrix3d& /*R_CB*/,
                             Eigen::MatrixXd* dpC_dT,
                             const Eigen::Matrix3d* /*R_CfCa*/,
                             const Eigen::Vector4d* /*ab1rho*/) {
    *dpC_dT = Eigen::MatrixXd();
  }

  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d& /*hpC*/,
                                 const Eigen::Matrix3d& /*R_CB*/,
                                 Eigen::MatrixXd* dpC_dT) {
    *dpC_dT = Eigen::MatrixXd();
  }

  static void toParamsInfo(const std::string /*delimiter*/,
                           std::string* extrinsic_format) {
    *extrinsic_format = "";
  }
  static void toParamsValueString(const okvis::kinematics::Transformation& /*T_SC*/,
                                  const std::string /*delimiter*/,
                                  std::string* extrinsic_string) {
    *extrinsic_string = "";
  }
  static void toParamValues(
      const okvis::kinematics::Transformation& /*T_SC*/,
      Eigen::VectorXd* /*extrinsic_opt_coeffs*/) {
    return;
  }
  template <class Scalar>
  static std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> get_T_BC(
      const okvis::kinematics::Transformation& T_BC_base,
      const Scalar* /*parameters*/) {
    return std::make_pair(T_BC_base.r().cast<Scalar>(), T_BC_base.q().cast<Scalar>());
  }
};

class Extrinsic_p_CB {
  // of T_BC only p_CB is variable.
 public:
  static const int kModelId = 1;
  static const size_t kNumParams = 3;
  static const size_t kGlobalDim = 3;
  static inline int getMinimalDim() { return kNumParams; }
  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double /*sigma_orientation*/) {
    return Eigen::MatrixXd::Identity(3, 3) *
           (sigma_translation * sigma_translation);
  }
  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d& /*pC*/,
                             const Eigen::Matrix3d& /*R_CB*/,
                             Eigen::MatrixXd* dpC_dT,
                             const Eigen::Matrix3d* R_CfCa,
                             const Eigen::Vector4d* ab1rho) {
    *dpC_dT = (*ab1rho)[3] * (Eigen::Matrix3d::Identity() - (*R_CfCa));
  }

  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d& hpC,
                             const Eigen::Matrix3d& /*R_CB*/,
                             Eigen::MatrixXd* dpC_dT) {
      *dpC_dT = Eigen::Matrix3d::Identity() * hpC[3];
  }

  static void dhC_dExtrinsic_HPP(const Eigen::Matrix<double, 4, 1>& hpC,
                 const Eigen::Matrix<double, 3, 3>& /*R_CB*/,
                 Eigen::Matrix<double, 4, kNumParams>* dhC_deltaTBC) {
    dhC_deltaTBC->topLeftCorner<3, 3>() = Eigen::Matrix3d::Identity() * hpC[3];
    dhC_deltaTBC->row(3).setZero();
  }

  // the error state is $\delta p_B^C$ or $\delta p_S^C$
  static void updateState(const Eigen::Vector3d& r, const Eigen::Quaterniond& q,
                          const Eigen::VectorXd& delta,
                          Eigen::Vector3d* r_delta,
                          Eigen::Quaterniond* q_delta) {
    *r_delta = r - q * delta;
    *q_delta = q;
  }

  template <typename Scalar>
  static void oplus(
      const Scalar* const deltaT_BC,
      std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* T_BC) {
    Eigen::Map<const Eigen::Matrix<Scalar, 3, 1>> delta_t(deltaT_BC);
    T_BC->first -= T_BC->second * delta_t;
  }

  static void toParamsInfo(const std::string delimiter,
                           std::string* extrinsic_format) {
    *extrinsic_format =
        "p_BC_B_x[m]" + delimiter + "p_BC_B_y" + delimiter + "p_BC_B_z";
  }
  static void toParamsValueString(const okvis::kinematics::Transformation& T_BC,
                                  const std::string delimiter,
                                  std::string* extrinsic_string) {
    Eigen::Vector3d r = T_BC.q().conjugate() * (-T_BC.r());
    std::stringstream ss;
    ss << r[0] << delimiter << r[1] << delimiter << r[2];
    *extrinsic_string = ss.str();
  }
  static void toParamValues(
      const okvis::kinematics::Transformation& T_BC,
      Eigen::VectorXd* extrinsic_opt_coeffs) {
    *extrinsic_opt_coeffs = T_BC.q().conjugate() * (-T_BC.r());
  }

  template <class Scalar>
  static std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> get_T_BC(
      const okvis::kinematics::Transformation& T_BC_base,
      const Scalar* parameters) {
    Eigen::Matrix<Scalar, 3, 1> t_CB_C(parameters[0], parameters[1], parameters[2]);
    Eigen::Quaternion<Scalar> q_BC = T_BC_base.q().cast<Scalar>();
    return std::make_pair(q_BC * (-t_CB_C), q_BC);
  }
};

class Extrinsic_p_BC_q_BC {
  // T_BC is represented by p_BC and R_BC in the states.
 public:
  static const int kModelId = 2;
  static const size_t kNumParams = 6;
  static const size_t kGlobalDim = 7;
  static inline int getMinimalDim() { return kNumParams; }
  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double sigma_orientation) {
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.topLeftCorner<3, 3>() *= (sigma_translation * sigma_translation);
    cov.bottomRightCorner<3, 3>() *= (sigma_orientation * sigma_orientation);
    return cov;
  }

  /**
   * @brief dpC_dExtrinsic_AIDP anchored inverse depth
   * @param pC pC = (T_BC^{-1} * T_WB_{f_i}^{-1} * T_WB_a * T_BC * [a, b, 1, \rho]^T)_{1:3}
   *     R_BC = exp(\delta\theta) \hat{R}_BC
   *     t_BC = \delta t + \hat{t}_BC
   * @param R_CB
   * @param dpC_dT
   * @param R_CfCa
   * @param ab1rho
   */
  static void dpC_dExtrinsic_AIDP(const Eigen::Vector3d& pC,
                                  const Eigen::Matrix3d& R_CB,
                                  Eigen::MatrixXd* dpC_dT,
                                  const Eigen::Matrix3d* R_CfCa,
                                  const Eigen::Vector4d* ab1rho) {
    dpC_dT->resize(3, 6);
    dpC_dT->block<3, 3>(0, 0) =
        ((*R_CfCa) - Eigen::Matrix3d::Identity()) * R_CB * (*ab1rho)[3];
    dpC_dT->block<3, 3>(0, 3) =
        (kinematics::crossMx(pC) -
         (*R_CfCa) * kinematics::crossMx(ab1rho->head<3>())) *
        R_CB;
  }

  /**
   * @brief dpC_dExtrinsic_HPP homogeneous point
   * @param hpC hpC = T_BC^{-1} * T_WB_{f_i}^{-1} * [x,y,z,w]_W^T
   *     hpC = [pC, w]
   *     R_BC = exp(\delta\theta) \hat{R}_BC
   *     t_BC = \delta t + \hat{t}_BC
   * @param R_CB
   * @param dpC_dT 3x6
   */
  static void dpC_dExtrinsic_HPP(const Eigen::Vector4d& hpC,
                                 const Eigen::Matrix3d& R_CB,
                                 Eigen::MatrixXd* dpC_dT) {
    dpC_dT->resize(3, 6);
    dpC_dT->block<3, 3>(0, 0) = -R_CB * hpC[3];
    dpC_dT->block<3, 3>(0, 3) = kinematics::crossMx(hpC.head<3>()) * R_CB;
  }

  // see dpC_dExtrinsic_HPP
  static void dhC_dExtrinsic_HPP(const Eigen::Matrix<double, 4, 1>& hpC,
                 const Eigen::Matrix<double, 3, 3>& R_CB,
                 Eigen::Matrix<double, 4, kNumParams>* dhC_deltaTBC) {
    dhC_deltaTBC->block<3, 3>(0, 0) = -R_CB * hpC[3];
    dhC_deltaTBC->block<3, 3>(0, 3) = kinematics::crossMx(hpC.head<3>()) * R_CB;
    dhC_deltaTBC->row(3).setZero();
  }

  static void updateState(const Eigen::Vector3d& r, const Eigen::Quaterniond& q,
                          const Eigen::VectorXd& delta,
                          Eigen::Vector3d* r_delta,
                          Eigen::Quaterniond* q_delta) {
    Eigen::Vector3d deltaAlpha = delta.segment<3>(3);   
    *r_delta = r + delta.head<3>();
    *q_delta = okvis::ceres::expAndTheta(deltaAlpha) * q;
    q_delta->normalize();
  }

  template <typename Scalar>
  static void oplus(
      const Scalar* const deltaT_BC,
      std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>>* T_BC) {
    Eigen::Map<const Eigen::Matrix<Scalar, 6, 1>> deltaT_BCe(deltaT_BC);
    T_BC->first += deltaT_BCe.template head<3>();
    Eigen::Matrix<Scalar, 3, 1> omega = deltaT_BCe.template tail<3>();
    Eigen::Quaternion<Scalar> dqSC = okvis::ceres::expAndTheta(omega);
    T_BC->second = dqSC * T_BC->second;
  }

  static void toParamsInfo(const std::string delimiter,
                           std::string* extrinsic_format) {
    *extrinsic_format = "p_BC_S_x[m]" + delimiter + "p_BC_S_y" + delimiter +
                        "p_BC_S_z" + delimiter + "q_BC_x" + delimiter +
                        "q_BC_y" + delimiter + "q_BC_z" + delimiter + "q_BC_w";
  }
  static void toParamsValueString(const okvis::kinematics::Transformation& T_BC,
                                  const std::string delimiter,
                                  std::string* extrinsic_string) {
    Eigen::Vector3d r = T_BC.r();
    Eigen::Quaterniond q = T_BC.q();
    std::stringstream ss;
    ss << r[0] << delimiter << r[1] << delimiter << r[2];
    ss << delimiter << q.x() << delimiter << q.y() << delimiter << q.z() << delimiter << q.w();
    *extrinsic_string = ss.str();
  }
  static void toParamValues(
      const okvis::kinematics::Transformation& T_BC,
      Eigen::VectorXd* extrinsic_opt_coeffs) {
    *extrinsic_opt_coeffs = T_BC.coeffs();
  }

  template <class Scalar>
  static std::pair<Eigen::Matrix<Scalar, 3, 1>, Eigen::Quaternion<Scalar>> get_T_BC(
      const okvis::kinematics::Transformation& /*T_BC_base*/,
      const Scalar* parameters) {
    Eigen::Matrix<Scalar, 3, 1> t_BC_B(parameters[0], parameters[1], parameters[2]);
    Eigen::Quaternion<Scalar> q_BC(parameters[6], parameters[3], parameters[4],
                                   parameters[5]);
    return std::make_pair(t_BC_B, q_BC);
  }
};

#ifndef EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASES          \
  EXTRINSIC_MODEL_CASE(ExtrinsicFixed) \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_CB) \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_BC_q_BC)
#endif

inline int ExtrinsicModelGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::getMinimalDim();

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline Eigen::MatrixXd ExtrinsicModelInitCov(int model_id,
                                             double sigma_translation,
                                             double sigma_orientation) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::initCov(sigma_translation, sigma_orientation);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline int ExtrinsicModelNameToId(std::string extrinsic_opt_rep) {
  std::transform(extrinsic_opt_rep.begin(), extrinsic_opt_rep.end(),
                 extrinsic_opt_rep.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (extrinsic_opt_rep.compare("P_CB") == 0) {
    return Extrinsic_p_CB::kModelId;
  } else if (extrinsic_opt_rep.compare("P_BC_R_BC") == 0) {
    return Extrinsic_p_BC_q_BC::kModelId;
  } else {
    return ExtrinsicFixed::kModelId;
  }
}

inline void ExtrinsicModelUpdateState(int model_id, const Eigen::Vector3d& r,
                                      const Eigen::Quaterniond& q,
                                      const Eigen::VectorXd& delta,
                                      Eigen::Vector3d* r_delta,
                                      Eigen::Quaterniond* q_delta) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::updateState(r, q, delta, r_delta, q_delta);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModel_dpC_dExtrinsic_AIDP(int model_id, const Eigen::Vector3d& pC,
                                               const Eigen::Matrix3d& R_CB,
                                               Eigen::MatrixXd* dpC_dT,
                                               const Eigen::Matrix3d* R_CfCa,
                                               const Eigen::Vector4d* ab1rho) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::dpC_dExtrinsic_AIDP(pC, R_CB, dpC_dT, R_CfCa, ab1rho);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModel_dpC_dExtrinsic_HPP(int model_id, const Eigen::Vector4d& hpC,
                                              const Eigen::Matrix3d& R_CB,
                                              Eigen::MatrixXd* dpC_dT) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::dpC_dExtrinsic_HPP(hpC, R_CB, dpC_dT);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModelToParamsInfo(
    int model_id, const std::string delimiter, std::string* extrinsic_format) {
    switch (model_id) {
  #define MODEL_CASES EXTRINSIC_MODEL_CASES
  #define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
    case ExtrinsicModel::kModelId:             \
      return ExtrinsicModel::toParamsInfo(delimiter, extrinsic_format);

      MODEL_SWITCH_CASES

  #undef EXTRINSIC_MODEL_CASE
  #undef MODEL_CASES
    }
}

inline void ExtrinsicModelToParamsValueString(
    int model_id, const okvis::kinematics::Transformation& T_BC,
    const std::string delimiter, std::string* extrinsic_string) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                    \
  case ExtrinsicModel::kModelId:                                \
    return ExtrinsicModel::toParamsValueString(T_BC, delimiter, \
                                               extrinsic_string);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ExtrinsicModelToParamValues(
    int model_id, const okvis::kinematics::Transformation& T_BC,
    Eigen::VectorXd* extrinsic_opt_coeffs) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel)                    \
  case ExtrinsicModel::kModelId:                                \
    return ExtrinsicModel::toParamValues(                       \
        T_BC, extrinsic_opt_coeffs);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
  }
}

}  // namespace okvis
#endif  // INCLUDE_MSCKF_EXTRINSIC_MODELS_HPP_
