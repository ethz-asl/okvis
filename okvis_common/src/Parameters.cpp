#include <okvis/Parameters.hpp>
#include <unordered_map>

namespace okvis {
ExtrinsicsEstimationParameters::ExtrinsicsEstimationParameters()
    : sigma_absolute_translation(0.0),
      sigma_absolute_orientation(0.0),
      sigma_c_relative_translation(0.0),
      sigma_c_relative_orientation(0.0),
      sigma_focal_length(0.0),
      sigma_principal_point(0.0),
      sigma_td(0.0),
      sigma_tr(0.0) {}

ExtrinsicsEstimationParameters::ExtrinsicsEstimationParameters(
    double sigma_absolute_translation, double sigma_absolute_orientation,
    double sigma_c_relative_translation, double sigma_c_relative_orientation)
    : sigma_absolute_translation(sigma_absolute_translation),
      sigma_absolute_orientation(sigma_absolute_orientation),
      sigma_c_relative_translation(sigma_c_relative_translation),
      sigma_c_relative_orientation(sigma_c_relative_orientation) {}

ImuParameters::ImuParameters()
    : T_BS(),
      a_max(200.0),
      g_max(10),
      sigma_g_c(1.2e-3),
      sigma_a_c(8e-3),
      sigma_bg(0.03),
      sigma_ba(0.1),
      sigma_gw_c(4e-6),
      sigma_aw_c(4e-5),
      tau(3600.0),
      g(9.80665),
      g0(0, 0, 0),
      a0(0, 0, 0),
      rate(100),
      sigma_TGElement(5e-3),
      sigma_TSElement(1e-3),
      sigma_TAElement(5e-3),
      model_type("BG_BA_TG_TS_TA") {
  Eigen::Matrix<double, 9, 1> eye;
  eye << 1, 0, 0, 0, 1, 0, 0, 0, 1;
  Tg0 = eye;
  Ts0.setZero();
  Ta0 = eye;
}

EstimatorAlgorithm EstimatorAlgorithmNameToId(std::string description) {
  std::transform(description.begin(), description.end(), description.begin(),
                 ::toupper);
  std::unordered_map<std::string, EstimatorAlgorithm> descriptionToId{
      {"OKVIS", EstimatorAlgorithm::OKVIS},
      {"GENERAL", EstimatorAlgorithm::General},
      {"PRIORLESS", EstimatorAlgorithm::Priorless},
      {"MSCKF", EstimatorAlgorithm::MSCKF},
      {"TFVIO", EstimatorAlgorithm::TFVIO},
      {"INVARIANTEKF", EstimatorAlgorithm::InvariantEKF},
      {"SLIDINGWINDOWSMOOTHER", EstimatorAlgorithm::SlidingWindowSmoother}};

  auto iter = descriptionToId.find(description);
  if (iter == descriptionToId.end()) {
    return EstimatorAlgorithm::OKVIS;
  } else {
    return iter->second;
  }
}

struct EstimatorAlgorithmHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

std::string EstimatorAlgorithmIdToName(EstimatorAlgorithm id) {
  std::unordered_map<EstimatorAlgorithm, std::string, EstimatorAlgorithmHash>
      idToDescription{{EstimatorAlgorithm::OKVIS, "OKVIS"},
                      {EstimatorAlgorithm::General, "General"},
                      {EstimatorAlgorithm::Priorless, "Priorless"},
                      {EstimatorAlgorithm::MSCKF, "MSCKF"},
                      {EstimatorAlgorithm::TFVIO, "TFVIO"},
                      {EstimatorAlgorithm::InvariantEKF, "InvariantEKF"},
                      {EstimatorAlgorithm::SlidingWindowSmoother, "SlidingWindowSmoother"}};
  auto iter = idToDescription.find(id);
  if (iter == idToDescription.end()) {
    return "OKVIS";
  } else {
    return iter->second;
  }
}

Optimization::Optimization()
    : max_iterations(10),
      min_iterations(3),
      timeLimitForMatchingAndOptimization(0.035),
      timeReserve(0.005),
      detectionThreshold(40),
      useMedianFilter(false),
      detectionOctaves(0),
      maxNoKeypoints(400),
      numKeyframes(5),
      numImuFrames(3),
      keyframeInsertionOverlapThreshold(0.6),
      keyframeInsertionMatchingRatioThreshold(0.2),
      algorithm(EstimatorAlgorithm::OKVIS),
      translationThreshold(0.4),
      rotationThreshold(0.2618),
      trackingRateThreshold(0.5),
      minTrackLength(3u),
      triangulationTranslationThreshold(-1.0),
      triangulationMaxDepth(1000.0),
      useEpipolarConstraint(false),
      cameraObservationModelId(0) {}

FrontendOptions::FrontendOptions(bool initWithoutEnoughParallax,
                                 bool stereoWithEpipolarCheck,
                                 double epipolarDistanceThresh,
                                 int featureTrackingApproach)
    : initializeWithoutEnoughParallax(initWithoutEnoughParallax),
      stereoMatchWithEpipolarCheck(stereoWithEpipolarCheck),
      epipolarDistanceThreshold(epipolarDistanceThresh),
      featureTrackingMethod(featureTrackingApproach) {}

PoseGraphOptions::PoseGraphOptions()
    : maxOdometryConstraintForAKeyframe(3) {}

PointLandmarkOptions::PointLandmarkOptions()
    : landmarkModelId(0), anchorAtObservationTime(false) {}

PointLandmarkOptions::PointLandmarkOptions(int lmkModelId, bool anchorAtObsTime)
    : landmarkModelId(lmkModelId), anchorAtObservationTime(anchorAtObsTime) {}

}  // namespace okvis
