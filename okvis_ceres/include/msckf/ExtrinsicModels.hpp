#ifndef INCLUDE_MSCKF_EXTRINSIC_MODELS_HPP_
#define INCLUDE_MSCKF_EXTRINSIC_MODELS_HPP_

#include <Eigen/Core>
#include <msckf/ModelSwitch.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <okvis/kinematics/operators.hpp>

namespace okvis {
class ExtrinsicFixed {
 public:
  static const int kModelId = 0;

  static inline int getMinimalDim() { return 0; }
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
  static void toParamsInfo(const std::string delimiter,
                           std::string* extrinsic_format) {
    *extrinsic_format = "p_SC_S_x[m]" + delimiter + "p_SC_S_y" + delimiter +
                        "p_SC_S_z" + delimiter + "q_SC_x" + delimiter +
                        "q_SC_y" + delimiter + "q_SC_z" + delimiter + "q_SC_w";
  }
};

class Extrinsic_p_CS {
  // of T_SC only p_CS is variable
 public:
  static const int kModelId = 1;
  static inline int getMinimalDim() { return 3; }
  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double /*sigma_orientation*/) {
    return Eigen::MatrixXd::Identity(3, 3) *
           (sigma_translation * sigma_translation);
  }
  static void dpC_dExtrinsic(const Eigen::Vector3d& /*pC*/,
                             const Eigen::Matrix3d& /*R_CB*/,
                             Eigen::MatrixXd* dpC_dT,
                             const Eigen::Matrix3d* R_CfCa,
                             const Eigen::Vector4d* ab1rho) {
    if (ab1rho != nullptr) {
      *dpC_dT = (*ab1rho)[3] * (Eigen::Matrix3d::Identity() - (*R_CfCa));
    } else {
      *dpC_dT = Eigen::Matrix3d::Identity();
    }
  }
  // the error state is $\delta p_B^C$ or $\delta p_S^C$
  static void updateState(const Eigen::Vector3d& r, const Eigen::Quaterniond& q,
                          const Eigen::VectorXd& delta,
                          Eigen::Vector3d* r_delta,
                          Eigen::Quaterniond* q_delta) {
    *r_delta = r - q.toRotationMatrix() * delta;
    *q_delta = q;
  }
  static void toParamsInfo(const std::string delimiter,
                           std::string* extrinsic_format) {
    *extrinsic_format =
        "p_SC_S_x[m]" + delimiter + "p_SC_S_y" + delimiter + "p_SC_S_z";
  }
};

class Extrinsic_p_SC_q_SC {
  // T_SC is represented by p_SC and R_SC in the states
 public:
  static const int kModelId = 2;

  static inline int getMinimalDim() { return 6; }
  static inline Eigen::MatrixXd initCov(double sigma_translation,
                                        double sigma_orientation) {
    Eigen::Matrix<double, 6, 6> cov = Eigen::Matrix<double, 6, 6>::Identity();
    cov.topLeftCorner<3, 3>() *= (sigma_translation * sigma_translation);
    cov.bottomRightCorner<3, 3>() *= (sigma_orientation * sigma_orientation);
    return cov;
  }
  // if ab1rho is not null, AIDP is used, then pC is actually \rho * pC
  // which is used in the projection function
  static void dpC_dExtrinsic(const Eigen::Vector3d& pC,
                             const Eigen::Matrix3d& R_CB,
                             Eigen::MatrixXd* dpC_dT,
                             const Eigen::Matrix3d* R_CfCa,
                             const Eigen::Vector4d* ab1rho) {
    if (ab1rho != nullptr) {
      *dpC_dT = Eigen::Matrix<double, 3, 6>();
      dpC_dT->block<3, 3>(0, 0) =
          ((*R_CfCa) - Eigen::Matrix3d::Identity()) * R_CB * (*ab1rho)[3];
      dpC_dT->block<3, 3>(0, 3) =
          (kinematics::crossMx(pC) -
           (*R_CfCa) * kinematics::crossMx(ab1rho->head<3>())) *
          R_CB;
    } else {
      *dpC_dT = Eigen::Matrix<double, 3, 6>();
      dpC_dT->block<3, 3>(0, 0) = -R_CB;
      dpC_dT->block<3, 3>(0, 3) = kinematics::crossMx(pC) * R_CB;
    }
  }
  static void updateState(const Eigen::Vector3d& r, const Eigen::Quaterniond& q,
                          const Eigen::VectorXd& delta,
                          Eigen::Vector3d* r_delta,
                          Eigen::Quaterniond* q_delta) {
    Eigen::Vector3d deltaAlpha = delta.segment<3>(3);
    Eigen::Vector4d dq;
    double halfnorm = 0.5 * deltaAlpha.norm();
    dq.head<3>() = okvis::kinematics::sinc(halfnorm) * 0.5 * deltaAlpha;
    dq[3] = cos(halfnorm);

    *r_delta = r + delta.head<3>();
    *q_delta = Eigen::Quaterniond(dq) * q;
    q_delta->normalize();
  }
  static void toParamsInfo(const std::string delimiter,
                           std::string* extrinsic_format) {
    *extrinsic_format = "p_SC_S_x[m]" + delimiter + "p_SC_S_y" + delimiter +
                        "p_SC_S_z" + delimiter + "q_SC_x" + delimiter +
                        "q_SC_y" + delimiter + "q_SC_z" + delimiter + "q_SC_w";
  }
};

#ifndef EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASES          \
  EXTRINSIC_MODEL_CASE(ExtrinsicFixed) \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_CS) \
  EXTRINSIC_MODEL_CASE(Extrinsic_p_SC_q_SC)
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
  if (extrinsic_opt_rep.compare("P_CS") == 0) {
    return Extrinsic_p_CS::kModelId;
  } else if (extrinsic_opt_rep.compare("P_SC_R_SC") == 0) {
    return Extrinsic_p_SC_q_SC::kModelId;
  } else {
    return ExtrinsicFixed::kModelId;
  }
}

inline void ExtrinsicModel_dpC_dExtrinsic(int model_id,
                                          const Eigen::Vector3d& pC,
                                          const Eigen::Matrix3d& R_CB,
                                          Eigen::MatrixXd* dpC_dT,
                                          const Eigen::Matrix3d* R_CfCa,
                                          const Eigen::Vector4d* ab1rho) {
  switch (model_id) {
#define MODEL_CASES EXTRINSIC_MODEL_CASES
#define EXTRINSIC_MODEL_CASE(ExtrinsicModel) \
  case ExtrinsicModel::kModelId:             \
    return ExtrinsicModel::dpC_dExtrinsic(pC, R_CB, dpC_dT, R_CfCa, ab1rho);

    MODEL_SWITCH_CASES

#undef EXTRINSIC_MODEL_CASE
#undef MODEL_CASES
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

}  // namespace okvis
#endif  // INCLUDE_MSCKF_EXTRINSIC_MODELS_HPP_
