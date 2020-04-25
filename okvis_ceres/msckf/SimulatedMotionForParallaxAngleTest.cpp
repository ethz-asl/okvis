#include "msckf/SimulatedMotionForParallaxAngleTest.hpp"
#include <map>

namespace simul {
std::ostream& operator<<(std::ostream& out, const MotionType value) {
  static std::map<MotionType, std::string> strings;
  if (strings.size() == 0) {
#define INSERT_ELEMENT(p) strings[p] = #p
    INSERT_ELEMENT(Sideways);
    INSERT_ELEMENT(Sideways_R_WC);
    INSERT_ELEMENT(Forward);
    INSERT_ELEMENT(Backward);
    INSERT_ELEMENT(Mixed);
    INSERT_ELEMENT(AssociateObserver);
    INSERT_ELEMENT(Rotation);
    INSERT_ELEMENT(AntiRotation);
    INSERT_ELEMENT(ForwardWithCenterPoint);
#undef INSERT_ELEMENT
  }
  return out << strings[value];
}

Eigen::Matrix3d RotY(double theta) {
  double ct = std::cos(theta);
  double st = std::sin(theta);
  Eigen::Matrix3d roty;
  roty << ct, 0, st, 0, 1, 0, -st, 0, ct;
  return roty;
}

SimulatedMotionForParallaxAngleTest::SimulatedMotionForParallaxAngleTest(
    MotionType motion, bool addOutlierObservation)
    : motionType_(motion) {
  srand((unsigned int)time(0));  // comment this for deterministic behavior
  const double a = 1.0;
  const double root3 = std::sqrt(3.0);
  const double b = 2;
  T_WC_list_.reserve(8);
  observationsxy1_.reserve(8);
  anchorObservationIndices_.reserve(2);
  std::vector<bool> projectionStatus;
  projectionStatus.reserve(8);
  observationListStatus_ = Healthy;
  switch (motionType_) {
    case Sideways:
      simulateSidewaysMotion(a, addOutlierObservation);
      break;
    case Sideways_R_WC:
      pW_ << 8.41028, 4.59741, -3.42285;
      T_WCm_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(4, 0.8, 0), Eigen::Quaterniond(0.5, -0.5, 0.5, -0.5));
      T_WCa_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(4, 1.6, 0), Eigen::Quaterniond(0.5, -0.5, 0.5, -0.5));
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(4, 2.4, 0), Eigen::Quaterniond(0.5, -0.5, 0.5, -0.5));
      break;
    case Forward:
      pW_ << 0, -a, (2 + root3) * a;
      T_WCm_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
      T_WCa_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 2 * a), Eigen::Quaterniond::Identity());
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, a), Eigen::Quaterniond::Identity());
      break;
    case Backward:
      pW_ << 0, -a, (2 + root3) * a;
      T_WCm_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 2 * a), Eigen::Quaterniond::Identity());
      T_WCa_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, a), Eigen::Quaterniond::Identity());
      break;
    case ForwardWithCenterPoint:
      pW_ << 0, 0, (2 + root3) * a;
      T_WCm_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
      T_WCa_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 2 * a), Eigen::Quaterniond::Identity());
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(a, 0, a), Eigen::Quaterniond::Identity());
      break;
    case Rotation:
      pW_ << b * sin(15 * M_PI / 180), -b / root3, b * cos(15 * M_PI / 180);
      T_WCm_ = okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                                 Eigen::Quaterniond(RotY(0)));
      T_WCa_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(RotY(30 * M_PI / 180)));
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, a), Eigen::Quaterniond(RotY(15 * M_PI / 180)));
      break;
    case AntiRotation:
      pW_ << b * sin(15 * M_PI / 180), -b / root3, b * cos(15 * M_PI / 180);
      T_WCa_ = okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                                 Eigen::Quaterniond(RotY(0)));
      T_WCm_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(RotY(30 * M_PI / 180)));
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, a), Eigen::Quaterniond(RotY(15 * M_PI / 180)));
      break;
    case AssociateObserver:
      pW_ << a, -a, root3 * a;
      T_WCm_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
      T_WCa_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(2 * a, 0, 0),
          Eigen::Quaterniond(RotY(-30 * M_PI / 180)));
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(2 * a, 0, 0),
          Eigen::Quaterniond(RotY(-30 * M_PI / 180)));
    case Mixed:
    default:
      pW_ << a, -a, root3 * a;
      T_WCm_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
      T_WCa_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(2 * a, 0, 0),
          Eigen::Quaterniond(RotY(-30 * M_PI / 180)));
      T_WCj_ = okvis::kinematics::Transformation(
          Eigen::Vector3d(a, 0, 0), Eigen::Quaterniond(RotY(-15 * M_PI / 180)));
      break;
  }
  cosParallaxAngle_ = computeCosParallaxAngle();
  pap_ = LWF::ParallaxAnglePoint(pW_, cosParallaxAngle_);
  dfpaj_.initialize(T_WCm_, T_WCa_.r(), T_WCj_.r(), pap_);
  Nij_ = dfpaj_.evaluate();
}

void SimulatedMotionForParallaxAngleTest::dN_dp_WCmi(Eigen::Matrix3d* j) const {
  double h = 1e-6;
  Eigen::Matrix<double, 6, 1> delta;

  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = h;
    okvis::kinematics::Transformation T_WCm_delta = T_WCm_;
    T_WCm_delta.oplus(delta);
    msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(
        T_WCm_delta, T_WCa_.r(), T_WCj_.r(), pap_);
    Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
    j->col(i) = (Nij_delta - Nij_) / h;
  }
}

void SimulatedMotionForParallaxAngleTest::dN_dtheta_WCmi(
    Eigen::Matrix3d* j) const {
  double h = 1e-6;
  Eigen::Matrix<double, 6, 1> delta;

  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i + 3] = h;
    okvis::kinematics::Transformation T_WCm_delta = T_WCm_;
    T_WCm_delta.oplus(delta);
    msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(
        T_WCm_delta, T_WCa_.r(), T_WCj_.r(), pap_);
    Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
    j->col(i) = (Nij_delta - Nij_) / h;
  }
}

void SimulatedMotionForParallaxAngleTest::dN_dp_WCai(Eigen::Matrix3d* j) const {
  double h = 1e-6;
  Eigen::Matrix<double, 6, 1> delta;
  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = h;
    okvis::kinematics::Transformation T_WCa_delta = T_WCa_;
    T_WCa_delta.oplus(delta);
    msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(
        T_WCm_, T_WCa_delta.r(), T_WCj_.r(), pap_);
    Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
    j->col(i) = (Nij_delta - Nij_) / h;
  }
}

void SimulatedMotionForParallaxAngleTest::dN_dp_WCtij(
    Eigen::Matrix3d* j) const {
  double h = 1e-6;
  Eigen::Matrix<double, 6, 1> delta;
  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = h;
    okvis::kinematics::Transformation T_WCj_delta = T_WCj_;
    T_WCj_delta.oplus(delta);
    msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(
        T_WCm_, T_WCa_.r(), T_WCj_delta.r(), pap_);
    Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
    j->col(i) = (Nij_delta - Nij_) / h;
  }
}

void SimulatedMotionForParallaxAngleTest::dN_dni(
    Eigen::Matrix<double, 3, 2>* j) const {
  double h = 1e-6;
  Eigen::Vector2d delta;
  for (int i = 0; i < 2; ++i) {
    delta.setZero();
    delta[i] = h;
    LWF::ParallaxAnglePoint papDelta = pap_;
    papDelta.n_.boxPlus(delta, papDelta.n_);
    msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_, T_WCa_.r(),
                                                          T_WCj_.r(), papDelta);
    Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
    j->col(i) = (Nij_delta - Nij_) / h;
  }
}

void SimulatedMotionForParallaxAngleTest::dN_dthetai(Eigen::Vector3d* j) const {
  double h = 1e-6;
  LWF::ParallaxAnglePoint papDelta = pap_;
  Eigen::Matrix<double, 1, 1> delta;
  delta[0] = h;
  papDelta.theta_.boxPlus(delta, papDelta.theta_);
  msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_, T_WCa_.r(),
                                                        T_WCj_.r(), papDelta);
  Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
  *j = (Nij_delta - Nij_) / h;
}

double SimulatedMotionForParallaxAngleTest::computeCosParallaxAngle(
    const Eigen::Vector3d& pW, const Eigen::Vector3d& t_WCm,
    const Eigen::Vector3d& t_WCa) const {
  return (pW - t_WCm).dot(pW - t_WCa) /
         ((pW - t_WCm).norm() * (pW - t_WCa).norm());
}

bool SimulatedMotionForParallaxAngleTest::computeObservationXY1(
    const Eigen::Vector3d pW, const okvis::kinematics::Transformation& T_WC,
    Eigen::Vector3d* xy1, bool addNoise) {
  Eigen::Vector3d pC = T_WC.C().transpose() * (pW - T_WC.r());
  *xy1 = pC / pC[2];
  if (addNoise) {
    xy1->head<2>() += Eigen::Vector2d::Random() / 600;
  }
  if (pC[2] < 0.1) {
    return false;
  }
  if (std::fabs((*xy1)[0]) > 1.5 || std::fabs((*xy1)[1]) > 1.2) {
    return false;
  }
  return true;
}

void SimulatedMotionForParallaxAngleTest::simulateSidewaysMotion(
    double a, bool addOutlier) {
  const double root3 = std::sqrt(3.0);
  std::vector<bool> projectionStatus;
  projectionStatus.reserve(8);
  pW_ << a, -a, root3 * a;
  T_WCm_ = okvis::kinematics::Transformation(Eigen::Vector3d(0, 0, 0),
                                             Eigen::Quaterniond::Identity());
  T_WCa_ = okvis::kinematics::Transformation(Eigen::Vector3d(2 * a, 0, 0),
                                             Eigen::Quaterniond::Identity());
  T_WCj_ = okvis::kinematics::Transformation(Eigen::Vector3d(a, 0, 0),
                                             Eigen::Quaterniond::Identity());
  T_WC_list_.push_back(T_WCm_);
  T_WC_list_.push_back(T_WCa_);
  T_WC_list_.push_back(T_WCj_);
  int augmentedObservations = 4;
  for (int j = 0; j < augmentedObservations; ++j) {
    T_WC_list_.emplace_back(Eigen::Vector3d(a + 0.4 * a * (j + 1), 0, 0),
                            Eigen::Quaterniond::Identity());
  }
  for (size_t j = 0; j < T_WC_list_.size(); ++j) {
    Eigen::Vector3d xy1;
    bool projectOk = computeObservationXY1(pW_, T_WC_list_.at(j), &xy1, true);
    observationsxy1_.push_back(xy1);
    projectionStatus.push_back(projectOk);
  }
  if (!projectionStatus[0]) {
    observationListStatus_ = MainAnchorFailed;
  }
  if (!projectionStatus[1]) {
    observationListStatus_ = AssociateAnchorFailed;
  }
  msckf::removeUnsetMatrices<okvis::kinematics::Transformation>(
      &T_WC_list_, projectionStatus);
  msckf::removeUnsetMatrices<Eigen::Vector3d>(&observationsxy1_,
                                              projectionStatus);
  if (addOutlier) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(2, T_WC_list_.size() - 1);
    int outlierIndex = dis(gen);
    std::cout << "real " << observationsxy1_[outlierIndex].transpose()
              << std::endl;
    observationsxy1_[outlierIndex] = Eigen::Vector3d::Random();
    observationsxy1_[outlierIndex][2] = 1.0;
    std::cout << "outlier " << observationsxy1_[outlierIndex].transpose()
              << std::endl;
  }
  anchorObservationIndices_.push_back(0u);
  anchorObservationIndices_.push_back(1u);
}
}  // namespace simul
