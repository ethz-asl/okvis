#ifndef INCLUDE_MSCKF_PARALLAX_ANGLE_POINT_HPP_
#define INCLUDE_MSCKF_PARALLAX_ANGLE_POINT_HPP_

#include <iostream>
#include <random>
#include <Eigen/Geometry>
#include <okvis/kinematics/Transformation.hpp> // for optimizing a PAP point.

namespace LWF {
typedef Eigen::Quaterniond QPD;
typedef Eigen::MatrixXd MXD;

/*!
 * \brief Gets a skew-symmetric matrix from a (column) vector
 * \param   vec 3x1-matrix (column vector)
 * \return skew   3x3-matrix
 */
template <typename PrimType_>
inline static Eigen::Matrix<PrimType_, 3, 3> skewVector(
    const Eigen::Matrix<PrimType_, 3, 1>& vec) {
  Eigen::Matrix<PrimType_, 3, 3> mat;
  mat << 0, -vec(2), vec(1), vec(2), 0, -vec(0), -vec(1), vec(0), 0;
  return mat;
}

template <typename DERIVED, typename GET, unsigned int D, unsigned int E = D>
class ElementBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  ElementBase(){};
  virtual ~ElementBase(){};
  static const unsigned int D_ = D;
  static const unsigned int E_ = E;
  typedef Eigen::Matrix<double, D_, 1> mtDifVec;
  typedef GET mtGet;
  std::string name_;
  virtual void boxPlus(const mtDifVec& vecIn, DERIVED& stateOut) const = 0;
  virtual void boxMinus(const DERIVED& stateIn, mtDifVec& vecOut) const = 0;
  virtual void boxMinusJac(const DERIVED& stateIn, MXD& matOut) const = 0;
  virtual void print() const = 0;
  virtual void setIdentity() = 0;
  virtual void setRandom(unsigned int& s) = 0;
  virtual void fix() = 0;
  static DERIVED Identity() {
    DERIVED identity;
    identity.setIdentity();
    return identity;
  }
  DERIVED& operator=(DERIVED other) {
    other.swap(*this);
    return *this;
  }
  virtual mtGet& get(unsigned int i) = 0;
  virtual const mtGet& get(unsigned int i) const = 0;
};

class NormalVectorElement
    : public ElementBase<NormalVectorElement, NormalVectorElement, 2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::Quaterniond q_;
  const Eigen::Vector3d e_x;
  const Eigen::Vector3d e_y;
  const Eigen::Vector3d e_z;
  NormalVectorElement() : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {}
  NormalVectorElement(const NormalVectorElement& other)
      : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {
    q_ = other.q_;
  }
  NormalVectorElement(const Eigen::Vector3d& vec)
      : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {
    setFromVector(vec);
  }
  NormalVectorElement(const Eigen::Quaterniond& q)
      : e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {
    q_ = q;
  }
  NormalVectorElement(double w, double x, double y, double z)
      : q_(w, x, y, z), e_x(1, 0, 0), e_y(0, 1, 0), e_z(0, 0, 1) {}
  virtual ~NormalVectorElement(){};
  Eigen::Vector3d getVec() const { return q_ * e_z; }
  Eigen::Vector3d getPerp1() const { return q_ * e_x; }
  Eigen::Vector3d getPerp2() const { return q_ * e_y; }
  NormalVectorElement& operator=(const NormalVectorElement& other) {
    q_ = other.q_;
    return *this;
  }
  static Eigen::Vector3d getRotationFromTwoNormals(
      const Eigen::Vector3d& a, const Eigen::Vector3d& b,
      const Eigen::Vector3d& a_perp) {
    const Eigen::Vector3d cross = a.cross(b);
    const double crossNorm = cross.norm();
    const double c = a.dot(b);
    const double angle = std::acos(c);
    if (crossNorm < 1e-6) {
      if (c > 0) {
        return cross;
      } else {
        return a_perp * M_PI;
      }
    } else {
      return cross * (angle / crossNorm);
    }
  }
  static Eigen::Vector3d getRotationFromTwoNormals(
      const NormalVectorElement& a, const NormalVectorElement& b) {
    return getRotationFromTwoNormals(a.getVec(), b.getVec(), a.getPerp1());
  }
  static Eigen::Matrix3d getRotationFromTwoNormalsJac(
      const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    const Eigen::Vector3d cross = a.cross(b);
    const double crossNorm = cross.norm();
    Eigen::Vector3d crossNormalized = cross / crossNorm;
    Eigen::Matrix3d crossNormalizedSqew = skewVector(crossNormalized);
    const double c = a.dot(b);
    const double angle = std::acos(c);
    if (crossNorm < 1e-6) {
      if (c > 0) {
        return -skewVector(b);
      } else {
        return Eigen::Matrix3d::Zero();
      }
    } else {
      return -1 / crossNorm *
             (crossNormalized * b.transpose() -
              (crossNormalizedSqew * crossNormalizedSqew * skewVector(b) *
               angle));
    }
  }
  static Eigen::Matrix3d getRotationFromTwoNormalsJac(
      const NormalVectorElement& a, const NormalVectorElement& b) {
    return getRotationFromTwoNormalsJac(a.getVec(), b.getVec());
  }
  void setFromVector(Eigen::Vector3d vec) {
    const double d = vec.norm();
    if (d > 1e-6) {
      vec = vec / d;
      Eigen::Vector3d rv = getRotationFromTwoNormals(e_z, vec, e_x);
      Eigen::AngleAxisd aa(rv.norm(), rv.normalized());
      q_ = Eigen::Quaterniond(aa);
    } else {
      q_.setIdentity();
    }
  }
  NormalVectorElement rotated(const QPD& q) const {
    return NormalVectorElement(q * q_);
  }
  NormalVectorElement inverted() const {
    Eigen::Quaterniond q(Eigen::AngleAxisd(M_PI, getPerp1()));
    return NormalVectorElement(q * q_);
  }
  void boxPlus(const mtDifVec& vecIn, NormalVectorElement& stateOut) const {
    Eigen::Vector3d Nu = vecIn(0) * getPerp1() + vecIn(1) * getPerp2();
    Eigen::Quaterniond q(Eigen::AngleAxisd(Nu.norm(), Nu.normalized()));
    stateOut.q_ = q * q_;
  }
  void boxMinus(const NormalVectorElement& stateIn, mtDifVec& vecOut) const {
    vecOut =
        stateIn.getN().transpose() * getRotationFromTwoNormals(stateIn, *this);
  }
  void boxMinusJac(const NormalVectorElement& stateIn, MXD& matOut) const {
    matOut = -stateIn.getN().transpose() *
             getRotationFromTwoNormalsJac(*this, stateIn) * this->getM();
  }
  void print() const { std::cout << getVec().transpose() << std::endl; }
  void setIdentity() { q_.setIdentity(); }
  void setRandom(unsigned int& s) {
    std::default_random_engine generator(s);
    std::normal_distribution<double> distribution(0.0, 1.0);
    q_.w() = distribution(generator);
    q_.x() = distribution(generator);
    q_.y() = distribution(generator);
    q_.z() = distribution(generator);
    q_.normalize();
    s++;
  }
  void fix() { q_.normalize(); }
  mtGet& get(unsigned int i = 0) {
    assert(i == 0);
    return *this;
  }
  const mtGet& get(unsigned int i = 0) const {
    assert(i == 0);
    return *this;
  }
  Eigen::Matrix<double, 3, 2> getM() const {
    Eigen::Matrix<double, 3, 2> M;
    M.col(0) = -getPerp2();
    M.col(1) = getPerp1();
    return M;
  }
  Eigen::Matrix<double, 3, 2> getN() const {
    Eigen::Matrix<double, 3, 2> M;
    M.col(0) = getPerp1();
    M.col(1) = getPerp2();
    return M;
  }
  const double* data() const { return q_.coeffs().data(); }
};

// Angle range [0, \pi]
class AngleElement : ElementBase<AngleElement, AngleElement, 1> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  void setFromCosine(double cosTheta) {
    cs_[0] = cosTheta;
    cs_[1] = std::sqrt(1 - cosTheta * cosTheta);
  }

  void set(const double* parameters) {
    cs_[0] = parameters[0];
    cs_[1] = parameters[1];
  }

  AngleElement() {}

  AngleElement(double cosTheta) { setFromCosine(cosTheta); }
  AngleElement(double ct, double st) : cs_(ct, st) {}
  virtual void boxPlus(const mtDifVec& vecIn, AngleElement& stateOut) const {
    double cd = std::cos(vecIn[0]);
    double sd = std::sin(vecIn[0]);
    double cSum = cs_[0] * cd - cs_[1] * sd;
    double sSum = cs_[1] * cd + sd * cs_[0];
    stateOut.cs_[0] = cSum;
    stateOut.cs_[1] = sSum;
  }

  virtual void boxMinus(const AngleElement& stateIn, mtDifVec& vecOut) const {
    double cSum = cs_[0] * stateIn.data()[0] + cs_[1] * stateIn.data()[1];
    double sSum = cs_[1] * stateIn.data()[0] - cs_[0] * stateIn.data()[1];

    vecOut[0] = std::atan2(sSum, cSum);
  }

  virtual void boxMinusJac(const AngleElement& /*stateIn*/, MXD& matOut) const {
    matOut = Eigen::Matrix<double, 1, 1>(1);
  }

  virtual void print() const {
    std::cout << getAngle() << std::endl;
  }
  virtual void setIdentity() {
    cs_[0] = 1;
    cs_[1] = 0;
  }
  virtual void setRandom(unsigned int& s) {
    std::default_random_engine generator(s);
    std::normal_distribution<double> distribution(0.0, 1.0);
    cs_[0] = distribution(generator);
    cs_[1] = distribution(generator);
    cs_.normalize();
    s++;
  }
  virtual void fix() { cs_.normalize(); }

  virtual mtGet& get(unsigned int /*i*/ = 0) { return *this; }

  virtual const mtGet& get(unsigned int /*i*/ = 0) const { return *this; }
  const double* data() const { return cs_.data(); }
  double getAngle() const { return std::acos(cs_[0]); }

 private:
  Eigen::Vector2d cs_;  // cos(\theta) sin(\theta). \theta is in the range of [0, \pi).
};

class ParallaxAnglePoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef Eigen::Matrix<double, 3, 1> mtDifVec;
  ParallaxAnglePoint() {}
  virtual ~ParallaxAnglePoint() {}
  /**
   * @brief ParallaxAnglePoint
   * @param bearing bearing vector in main anchor frame, not necessarily has
   * unit norm.
   * @param cosTheta cos(\theta)
   */
  ParallaxAnglePoint(const Eigen::Vector3d& bearing, double cosTheta)
      : theta_(cosTheta) {
    n_.setFromVector(bearing);
  }

  ParallaxAnglePoint(const NormalVectorElement& n, const AngleElement& theta)
      : n_(n), theta_(theta) {}

  ParallaxAnglePoint(double w, double x, double y, double z, double ct, double st)
      : n_(w, x, y, z), theta_(ct, st) {}

  void copy(std::vector<double>* parameters) const {
    parameters->resize(6);
    memcpy(parameters->data(), n_.data(), sizeof(double) * 4);
    memcpy(parameters->data() + 4, theta_.data(), sizeof(double) * 2);
  }

  void copy(Eigen::Matrix<double, 6, 1>* parameters) const {
    memcpy(parameters->data(), n_.data(), sizeof(double) * 4);
    memcpy(parameters->data() + 4, theta_.data(), sizeof(double) * 2);
  }

  void copy(double* parameters) const {
    memcpy(parameters, n_.data(), sizeof(double) * 4);
    memcpy(parameters + 4, theta_.data(), sizeof(double) * 2);
  }

  void set(const double* parameters) {
    n_ = NormalVectorElement(Eigen::Map<const Eigen::Quaterniond>(parameters));
    theta_.set(parameters + 4);
  }

  virtual void setRandom(unsigned int& s) {
    n_.setRandom(s);
    theta_.setRandom(s);
  }

  Eigen::Vector3d getVec() const { return n_.getVec(); }
  double getAngle() const { return theta_.getAngle(); }
  double cosTheta() const { return theta_.data()[0]; }
  double sinTheta() const { return theta_.data()[1]; }

  virtual void boxPlus(const mtDifVec& vecIn, ParallaxAnglePoint& stateOut) const {
    n_.boxPlus(vecIn.head<2>(), stateOut.n_);
    theta_.boxPlus(vecIn.tail<1>(), stateOut.theta_);
  }

  bool initializePosition(
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>&
          observationsxy1,
      const std::vector<
          okvis::kinematics::Transformation,
          Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WC_list,
      const std::vector<int>& anchorIndices);

  bool optimizePosition(
      const std::vector<Eigen::Vector3d,
                        Eigen::aligned_allocator<Eigen::Vector3d>>&
          observationsxy1,
      const std::vector<
          okvis::kinematics::Transformation,
          Eigen::aligned_allocator<okvis::kinematics::Transformation>>& T_WC_list,
      const std::vector<int>& anchorIndices);

  NormalVectorElement n_;
  AngleElement theta_;
};

}  // namespace LWF
#endif  // INCLUDE_MSCKF_PARALLAX_ANGLE_POINT_HPP_
