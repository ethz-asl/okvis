#include <gtest/gtest.h>

#include <Eigen/Core>
#include <msckf/DirectionFromParallaxAngleJacobian.hpp>
#include <msckf/VectorNormalizationJacobian.hpp>

TEST(VectorNormalizationJacobian, Evaluate) {
  Eigen::Vector3d vec;
  vec.setRandom();
  msckf::VectorNormalizationJacobian vnj(vec);
  Eigen::Matrix3d jac;
  vnj.dxi_dvec(&jac);
  Eigen::Vector3d vec_delta;
  Eigen::Vector3d delta;
  Eigen::Vector3d normalized_vec = vnj.normalized();
  Eigen::Matrix3d numericJac;
  double eps = 1e-6;
  for (int i = 0; i < 3; ++i) {
    delta.setZero();
    delta[i] = eps;
    vec_delta = vec + delta;
    numericJac.col(i) = vec_delta.normalized() - normalized_vec;
  }
  numericJac /= eps;
  EXPECT_LT((numericJac - jac).lpNorm<Eigen::Infinity>(), eps);
}

enum MotionType {
  Sideways = 0,
  Forward,
  Backward,
  Mixed,
  AssociateObserver,
  Rotation,
  AntiRotation,
  ForwardWithCenterPoint,
};

std::ostream& operator<<(std::ostream& out, const MotionType value){
    static std::map<MotionType, std::string> strings;
    if (strings.size() == 0){
#define INSERT_ELEMENT(p) strings[p] = #p
        INSERT_ELEMENT(Sideways);
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

class SimulatedMotionForParallaxAngleTest {
 public:
  SimulatedMotionForParallaxAngleTest(MotionType motion) : motionType_(motion) {
    srand((unsigned int)time(0));  // comment this for deterministic behavior
    const double a = 1.0;
    const double root3 = std::sqrt(3.0);
    const double b = 2;
    switch (motionType_) {
      case Sideways:
        pW_ << a, -a, root3 * a;
        T_WCm_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
        T_WCa_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(2 * a, 0, 0), Eigen::Quaterniond::Identity());
        T_WCj_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(a, 0, 0), Eigen::Quaterniond::Identity());
        break;
      case Forward:
        pW_ << 0, -a, (2+root3)* a;
        T_WCm_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
        T_WCa_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 2*a), Eigen::Quaterniond::Identity());
        T_WCj_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, a), Eigen::Quaterniond::Identity());
        break;
      case Backward:
        pW_ << 0, -a, (2+root3)* a;
        T_WCm_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 2*a), Eigen::Quaterniond::Identity());
        T_WCa_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
        T_WCj_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, a), Eigen::Quaterniond::Identity());
        break;
      case ForwardWithCenterPoint:
        pW_ << 0, 0, (2+root3)* a;
        T_WCm_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
        T_WCa_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 2*a), Eigen::Quaterniond::Identity());
        T_WCj_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(a, 0, a), Eigen::Quaterniond::Identity());
        break;
      case Rotation:
        pW_ << b * sin(15 * M_PI/180), -b/root3, b*cos(15 * M_PI/180);
        T_WCm_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(RotY(0)));
        T_WCa_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(RotY(30 * M_PI / 180)));
        T_WCj_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, a), Eigen::Quaterniond(RotY(15 * M_PI / 180)));
        break;
      case AntiRotation:
        pW_ << b * sin(15 * M_PI/180), -b/root3, b*cos(15 * M_PI/180);
        T_WCa_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond(RotY(0)));
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
            Eigen::Vector3d(2 * a, 0, 0), Eigen::Quaterniond(RotY(-30 * M_PI / 180)));
        T_WCj_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(2 * a, 0, 0), Eigen::Quaterniond(RotY(-30 * M_PI / 180)));
      case Mixed:
      default:
        pW_ << a, -a, root3 * a;
        T_WCm_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(0, 0, 0), Eigen::Quaterniond::Identity());
        T_WCa_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(2 * a, 0, 0), Eigen::Quaterniond(RotY(-30 * M_PI / 180)));
        T_WCj_ = okvis::kinematics::Transformation(
            Eigen::Vector3d(a, 0, 0), Eigen::Quaterniond(RotY(-15 * M_PI / 180)));
        break;
    }
    cosParallaxAngle_ = computeCosParallaxAngle();
    pap_ = LWF::ParallaxAnglePoint(pW_, cosParallaxAngle_);
    dfpaj_.initialize(T_WCm_, T_WCa_.r(), T_WCj_.r(), pap_);
    Nij_ = dfpaj_.evaluate();
  }

  void dN_dp_WCmi(Eigen::Matrix3d* j) const {
    double h = 1e-6;
    Eigen::Matrix<double, 6, 1> delta;

    for (int i = 0; i<3; ++i) {
      delta.setZero();
      delta[i] = h;
      okvis::kinematics::Transformation T_WCm_delta = T_WCm_;
      T_WCm_delta.oplus(delta);
      msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_delta, T_WCa_.r(), T_WCj_.r(), pap_);
      Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
      j->col(i) = (Nij_delta - Nij_)/h;
    }
  }

  void dN_dtheta_WCmi(Eigen::Matrix3d* j) const {
    double h = 1e-6;
    Eigen::Matrix<double, 6, 1> delta;

    for (int i = 0; i<3; ++i) {
      delta.setZero();
      delta[i + 3] = h;
      okvis::kinematics::Transformation T_WCm_delta = T_WCm_;
      T_WCm_delta.oplus(delta);
      msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_delta, T_WCa_.r(), T_WCj_.r(), pap_);
      Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
      j->col(i) = (Nij_delta - Nij_)/h;
    }
  }

  void dN_dp_WCai(Eigen::Matrix3d* j) const {
    double h = 1e-6;
    Eigen::Matrix<double, 6, 1> delta;
    for (int i = 0; i<3; ++i) {
      delta.setZero();
      delta[i] = h;
      okvis::kinematics::Transformation T_WCa_delta = T_WCa_;
      T_WCa_delta.oplus(delta);
      msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_, T_WCa_delta.r(), T_WCj_.r(), pap_);
      Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
      j->col(i) = (Nij_delta - Nij_)/h;
    }
  }

  void dN_dp_WCtij(Eigen::Matrix3d* j) const {
    double h = 1e-6;
    Eigen::Matrix<double, 6, 1> delta;
    for (int i = 0; i<3; ++i) {
      delta.setZero();
      delta[i] = h;
      okvis::kinematics::Transformation T_WCj_delta = T_WCj_;
      T_WCj_delta.oplus(delta);
      msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_, T_WCa_.r(), T_WCj_delta.r(), pap_);
      Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
      j->col(i) = (Nij_delta - Nij_)/h;
    }
  }

  void dN_dni(Eigen::Matrix<double, 3, 2>* j) const {
    double h = 1e-6;
    Eigen::Vector2d delta;
    for (int i=0; i<2; ++i) {
      delta.setZero();
      delta[i] = h;
      LWF::ParallaxAnglePoint papDelta = pap_;
      papDelta.n_.boxPlus(delta, papDelta.n_);
      msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_, T_WCa_.r(), T_WCj_.r(), papDelta);
      Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
      j->col(i) = (Nij_delta - Nij_)/h;
    }
  }

  void dN_dthetai(Eigen::Vector3d* j) const {
    double h = 1e-6;
    LWF::ParallaxAnglePoint papDelta = pap_;
    Eigen::Matrix<double, 1, 1> delta;
    delta[0] = h;
    papDelta.theta_.boxPlus(delta, papDelta.theta_);
    msckf::DirectionFromParallaxAngleJacobian dfpaj_delta(T_WCm_, T_WCa_.r(), T_WCj_.r(), papDelta);
    Eigen::Vector3d Nij_delta = dfpaj_delta.evaluate();
    *j = (Nij_delta - Nij_)/h;
  }

  msckf::DirectionFromParallaxAngleJacobian dfpaj_;

 private:
  double computeCosParallaxAngle() const {
    return computeCosParallaxAngle(pW_, T_WCm_.r(), T_WCa_.r());
  }

  double computeCosParallaxAngle(const Eigen::Vector3d& pW,
                                 const Eigen::Vector3d& t_WCm,
                                 const Eigen::Vector3d& t_WCa) const {
    return (pW - t_WCm).dot(pW - t_WCa) / ((pW - t_WCm).norm() * (pW - t_WCa).norm());
  }

  okvis::kinematics::Transformation T_WCm_;
  okvis::kinematics::Transformation T_WCa_;
  okvis::kinematics::Transformation T_WCj_;
  Eigen::Vector3d pW_;
  double cosParallaxAngle_;
  LWF::ParallaxAnglePoint pap_;
  Eigen::Vector3d Nij_;
  const MotionType motionType_;
};

#define DirectionFromParallaxAngleSubTest(METHOD, JACTYPE, EPS)            \
  TEST(DirectionFromParallaxAngle, METHOD) {                               \
    for (int i = MotionType::Sideways; i <= MotionType::AssociateObserver; \
         i++) {                                                            \
      MotionType mt = static_cast<MotionType>(i);                          \
      SimulatedMotionForParallaxAngleTest simulatedMotion(mt);             \
      JACTYPE numericJac;                                                  \
      simulatedMotion.METHOD(&numericJac);                                 \
      JACTYPE analyticJac;                                                 \
      simulatedMotion.dfpaj_.METHOD(&analyticJac);                         \
      EXPECT_LT((numericJac - analyticJac).lpNorm<Eigen::Infinity>(), EPS) \
          << #METHOD << ", " << mt << ", numeric\n"                        \
          << numericJac << "\nanalytic\n"                                  \
          << analyticJac;                                                  \
    }                                                                      \
  }

DirectionFromParallaxAngleSubTest(dN_dp_WCmi, Eigen::Matrix3d, 1e-6)

DirectionFromParallaxAngleSubTest(dN_dtheta_WCmi, Eigen::Matrix3d, 5e-6)

DirectionFromParallaxAngleSubTest(dN_dp_WCai, Eigen::Matrix3d, 1e-6)

DirectionFromParallaxAngleSubTest(dN_dp_WCtij, Eigen::Matrix3d, 1e-6)

DirectionFromParallaxAngleSubTest(dN_dthetai, Eigen::Vector3d, 1e-6)

typedef Eigen::Matrix<double,3,2> M32D;
DirectionFromParallaxAngleSubTest(dN_dni, M32D, 5e-6)

TEST(DirectionFromParallaxAngleAssociate, dN_dp_WCai) {
  SimulatedMotionForParallaxAngleTest simulatedMotion(MotionType::AssociateObserver);
  Eigen::Matrix3d numericJac;
  simulatedMotion.dN_dp_WCai(&numericJac);

  Eigen::Matrix3d analyticJac;
  simulatedMotion.dfpaj_.dN_dp_WCai(&analyticJac);

  Eigen::Matrix3d numericJac2;
  simulatedMotion.dN_dp_WCtij(&numericJac2);

  Eigen::Matrix3d analyticJac2;
  simulatedMotion.dfpaj_.dN_dp_WCtij(&analyticJac2);

  EXPECT_LT((numericJac + numericJac2 - analyticJac - analyticJac2).lpNorm<Eigen::Infinity>(), 1e-6);
}
