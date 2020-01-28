#ifndef INCLUDE_MSCKF_PROJ_PARAM_OPT_MODELS_HPP_
#define INCLUDE_MSCKF_PROJ_PARAM_OPT_MODELS_HPP_

#include <Eigen/Core>
#include <msckf/ModelSwitch.hpp>

namespace okvis {
//class ProjectionOptFixed {
// public:
//  static const int kModelId = 0;
//  static const size_t kNumParams = 0;
//  static inline int getMinimalDim() { return kNumParams; }
//  static void kneadIntrinsicJacobian(Eigen::Matrix2Xd* intrinsicJacobian) {
//      int resultCols = intrinsicJacobian->cols() - 4;
//      intrinsicJacobian->block(0, 0, 2, resultCols) =
//              intrinsicJacobian->block(0, 4, 2, resultCols);
//      intrinsicJacobian->conservativeResize(Eigen::NoChange, resultCols);
//  }
//  static void localToGlobal(const Eigen::VectorXd& /*local_opt_params*/,
//                            Eigen::VectorXd* /*global_proj_params*/) {
//    return;
//  }
//  static void globalToLocal(const Eigen::VectorXd& /*global_proj_params*/,
//                            Eigen::VectorXd* /*local_opt_params*/) {
//    return;
//  }
//  static Eigen::MatrixXd getInitCov(double /*sigma_focal_length*/,
//                                    double /*sigma_principal_point*/) {
//    return Eigen::MatrixXd();
//  }
//  static void toParamsInfo(const std::string /*delimiter*/, std::string* params_info) {
//    *params_info = "";
//  }
//};

class ProjectionOptFXY_CXY {
 public:
  static const int kModelId = 1;
  static const size_t kNumParams = 4;
  static inline int getMinimalDim() { return kNumParams; }

  static void localToGlobal(const Eigen::VectorXd& local_opt_params,
                            Eigen::VectorXd* global_proj_params) {
    global_proj_params->head<4>() = local_opt_params;
  }
  static void globalToLocal(const Eigen::VectorXd& global_proj_params,
                            Eigen::VectorXd* local_opt_params) {
    (*local_opt_params) = global_proj_params.head<4>();
  }

  static void kneadIntrinsicJacobian(Eigen::Matrix2Xd* /*intrinsicJacobian*/) {}
  static Eigen::MatrixXd getInitCov(double sigma_focal_length,
                                    double sigma_principal_point) {
    Eigen::MatrixXd covProjIntrinsics = Eigen::Matrix<double, 4, 4>::Identity();
    covProjIntrinsics.topLeftCorner<2, 2>() *= std::pow(sigma_focal_length, 2);
    covProjIntrinsics.bottomRightCorner<2, 2>() *=
        std::pow(sigma_principal_point, 2);
    return covProjIntrinsics;
  }
  static void toParamsInfo(const std::string delimiter, std::string* params_info) {
    *params_info = "fx[pixel]" + delimiter + "fy" + delimiter + "cx" + delimiter + "cy";
  }
};

class ProjectionOptFX_CXY {
 public:
  static const int kModelId = 2;
  static const size_t kNumParams = 3;
  static inline int getMinimalDim() { return kNumParams; }

  static void localToGlobal(const Eigen::VectorXd& local_opt_params,
                            Eigen::VectorXd* global_proj_params) {
    (*global_proj_params)(0) = local_opt_params[0];
    (*global_proj_params)(1) = local_opt_params[0];
    global_proj_params->segment<2>(2) = local_opt_params.segment<2>(1);
  }
  static void globalToLocal(const Eigen::VectorXd& global_proj_params,
                            Eigen::VectorXd* local_opt_params) {
    local_opt_params->resize(3, 1);
    (*local_opt_params)[0] = global_proj_params[0];
    local_opt_params->segment<2>(1) = global_proj_params.segment<2>(2);
  }
  static void kneadIntrinsicJacobian(Eigen::Matrix2Xd* intrinsicJac) {
    const int resultCols = intrinsicJac->cols() - 1;
    intrinsicJac->col(0) += intrinsicJac->col(1);
    intrinsicJac->block(0, 1, 2, resultCols - 1) =
        intrinsicJac->block(0, 2, 2, resultCols - 1);
    intrinsicJac->conservativeResize(Eigen::NoChange, resultCols);
  }
  static Eigen::MatrixXd getInitCov(double sigma_focal_length,
                                    double sigma_principal_point) {
    Eigen::MatrixXd covProjIntrinsics = Eigen::Matrix<double, 3, 3>::Identity();
    covProjIntrinsics(0, 0) *= std::pow(sigma_focal_length, 2);
    covProjIntrinsics.bottomRightCorner<2, 2>() *=
        std::pow(sigma_principal_point, 2);
    return covProjIntrinsics;
  }
  static void toParamsInfo(const std::string delimiter, std::string* params_info) {
    *params_info = "fx[pixel]" + delimiter + "cx" + delimiter + "cy";
  }
};

class ProjectionOptFX {
 public:
  static const int kModelId = 3;
  static const size_t kNumParams = 1;
  static inline int getMinimalDim() { return kNumParams; }

  static void localToGlobal(const Eigen::VectorXd& local_opt_params,
                            Eigen::VectorXd* global_proj_params) {
    (*global_proj_params)[0] = local_opt_params[0];
    (*global_proj_params)[1] = local_opt_params[0];
  }
  static void globalToLocal(const Eigen::VectorXd& global_proj_params,
                            Eigen::VectorXd* local_opt_params) {
    local_opt_params->resize(1, 1);
    (*local_opt_params)[0] = global_proj_params[0];
  }
  static void kneadIntrinsicJacobian(Eigen::Matrix2Xd* intrinsicJac) {
    const int resultCols = intrinsicJac->cols() - 3;
    intrinsicJac->col(0) += intrinsicJac->col(1);
    intrinsicJac->block(0, 1, 2, resultCols - 1) =
        intrinsicJac->block(0, 4, 2, resultCols - 1);
    intrinsicJac->conservativeResize(Eigen::NoChange, resultCols);
  }
  static Eigen::MatrixXd getInitCov(double sigma_focal_length,
                                    double /*sigma_principal_point*/) {
    Eigen::MatrixXd covProjIntrinsics =
        Eigen::Matrix<double, 1, 1>::Identity() *
        std::pow(sigma_focal_length, 2);
    return covProjIntrinsics;
  }
  static void toParamsInfo(const std::string /*delimiter*/, std::string* params_info) {
    *params_info = "fx[pixel]";
  }
};

#ifndef PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASES                \
  PROJ_OPT_MODEL_CASE(ProjectionOptFXY_CXY) \
  PROJ_OPT_MODEL_CASE(ProjectionOptFX_CXY)  \
  PROJ_OPT_MODEL_CASE(ProjectionOptFX)
#endif

inline int ProjectionOptGetMinimalDim(int model_id) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::getMinimalDim();

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
  return 0;
}

inline int ProjectionOptNameToId(std::string pinhole_opt_rep, bool* isFixed=nullptr) {
  std::transform(pinhole_opt_rep.begin(), pinhole_opt_rep.end(),
                 pinhole_opt_rep.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  if (isFixed) {
      *isFixed = false;
  }
  if (pinhole_opt_rep.compare("FXY_CXY") == 0) {
    return ProjectionOptFXY_CXY::kModelId;
  } else if (pinhole_opt_rep.compare("FX_CXY") == 0) {
    return ProjectionOptFX_CXY::kModelId;
  } else if (pinhole_opt_rep.compare("FX") == 0) {
    return ProjectionOptFX::kModelId;
  } else {
    if (isFixed) {
        *isFixed = true;
    }
    return ProjectionOptFXY_CXY::kModelId;
  }
}

inline void ProjectionOptKneadIntrinsicJacobian(
    int model_id, Eigen::Matrix2Xd* intrinsicJac) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::kneadIntrinsicJacobian(intrinsicJac);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
}

// apply estimated projection intrinsic params to the full param vector
// be careful global_proj_params may contain trailing distortion params
inline void ProjectionOptLocalToGlobal(int model_id,
                                       const Eigen::VectorXd& local_opt_params,
                                       Eigen::VectorXd* global_proj_params) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::localToGlobal(local_opt_params, global_proj_params);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
}

inline void ProjectionOptGlobalToLocal(
    int model_id, const Eigen::VectorXd& global_proj_params,
    Eigen::VectorXd* local_opt_params) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::globalToLocal(global_proj_params, local_opt_params);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
}

inline Eigen::MatrixXd ProjectionModelGetInitCov(int model_id,
                                                 double sigma_focal_length,
                                                 double sigma_principal_point) {
  switch (model_id) {
#define MODEL_CASES PROJ_OPT_MODEL_CASES
#define PROJ_OPT_MODEL_CASE(ProjOptModel) \
  case ProjOptModel::kModelId:            \
    return ProjOptModel::getInitCov(sigma_focal_length, sigma_principal_point);

    MODEL_SWITCH_CASES

#undef PROJ_OPT_MODEL_CASE
#undef MODEL_CASES
  }
  return Eigen::MatrixXd();
}

inline void ProjectionOptToParamsInfo(int model_id, const std::string delimiter, std::string* params_info) {
    switch (model_id) {
  #define MODEL_CASES PROJ_OPT_MODEL_CASES
  #define PROJ_OPT_MODEL_CASE(ProjOptModel) \
    case ProjOptModel::kModelId:            \
      return ProjOptModel::toParamsInfo(delimiter, params_info);

      MODEL_SWITCH_CASES

  #undef PROJ_OPT_MODEL_CASE
  #undef MODEL_CASES
    }
}

}  // namespace okvis
#endif  // INCLUDE_MSCKF_PROJ_PARAM_OPT_MODELS_HPP_
