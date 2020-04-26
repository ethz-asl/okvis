#include "gtest/gtest.h"

#include <msckf/InverseTransformPointJacobian.hpp>
#include <msckf/MultipleTransformPointJacobian.hpp>
#include <msckf/TransformPointJacobian.hpp>
#include <okvis/kinematics/Transformation.hpp>

class TransformPointJacobianTest : public ::testing::Test {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TransformPointJacobianTest() {}
    void SetUp() override {
      srand((unsigned int)time(0));  // comment this for deterministic behavior
      T_AB_.setRandom();
      hpB_.setRandom();
      tpj_.initialize(T_AB_, hpB_);
      hpA_ = tpj_.evaluate();
      computeNumericJacobians();
    }

    void computeNumericJacobians() {
      for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta;
        delta.setZero();
        delta(i) = eps;
        okvis::kinematics::Transformation T_AB_bar = T_AB_;
        T_AB_bar.oplus(delta);
        okvis::TransformPointJacobian tpj_bar(T_AB_bar, hpB_);
        Eigen::Vector4d hpA_bar = tpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
           (hpA_bar - hpA_) / eps;
        dhpA_dT_AB_numeric_.col(i) = ratio;
      }

      for (int i = 0; i < 4; ++i) {
        Eigen::Matrix<double, 4, 1> delta;
        delta.setZero();
        delta(i) = eps;
        Eigen::Vector4d hpB_bar = hpB_ + delta;

        okvis::TransformPointJacobian tpj_bar(T_AB_, hpB_bar);
        Eigen::Vector4d hpA_bar = tpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
            (hpA_bar - hpA_) / eps;
        dhpA_dhpB_numeric_.col(i) = ratio;
      }
    }

    void check() const {
      Eigen::Matrix<double, 4, 4> dhpA_dhpB;
      tpj_.dhpA_dhpB(&dhpA_dhpB);
      EXPECT_LT((dhpA_dhpB_numeric_ - dhpA_dhpB).lpNorm<Eigen::Infinity>(),
                tol);

      Eigen::Matrix<double, 4, 6> dhpA_dT_AB;
      tpj_.dhpA_dT_AB(&dhpA_dT_AB);
      EXPECT_LT((dhpA_dT_AB_numeric_ - dhpA_dT_AB).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpA_dT_AB_numeric_\n"
          << dhpA_dT_AB_numeric_ << "\ndhpA_dT_AB\n"
          << dhpA_dT_AB;
    }

    void checkAgainstMultipleTransformJacobian() const {
      okvis::MultipleTransformPointJacobian mtpj({T_AB_}, {1}, hpB_);
      Eigen::Matrix<double, 4, 4> dhpA_dhpB = mtpj.dp_dpoint();
      EXPECT_LT((dhpA_dhpB_numeric_ - dhpA_dhpB).lpNorm<Eigen::Infinity>(),
                tol);

      Eigen::Matrix<double, 4, 6> dhpA_dT_AB = mtpj.dp_dT(0u);
      EXPECT_LT((dhpA_dT_AB_numeric_ - dhpA_dT_AB).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpA_dT_AB_numeric_\n"
          << dhpA_dT_AB_numeric_ << "\ndhpA_dT_AB\n"
          << dhpA_dT_AB;
    }

    okvis::kinematics::Transformation T_AB_;
    Eigen::Vector4d hpA_;
    Eigen::Vector4d hpB_;
    okvis::TransformPointJacobian tpj_;

    Eigen::Matrix<double, 4, 4> dhpA_dhpB_numeric_;
    Eigen::Matrix<double, 4, 6> dhpA_dT_AB_numeric_;
    const double eps = 1e-6;
    const double tol = 1e-6;
};

TEST_F(TransformPointJacobianTest, dhp_dhp) {
  check();
}

TEST_F(TransformPointJacobianTest, MultipleTransform) {
  checkAgainstMultipleTransformJacobian();
}

class InverseTransformPointJacobianTest : public ::testing::Test {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    InverseTransformPointJacobianTest() {}
    void SetUp() override {
      srand((unsigned int)time(0));  // comment this for deterministic behavior
      T_AB_.setRandom();
      hpA_.setRandom();
      itpj_.initialize(T_AB_, hpA_);
      hpB_ = itpj_.evaluate();
      computeNumericJacobians();
    }

    void computeNumericJacobians() {
      for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta;
        delta.setZero();
        delta(i) = eps;
        okvis::kinematics::Transformation T_AB_bar = T_AB_;
        T_AB_bar.oplus(delta);
        okvis::InverseTransformPointJacobian itpj_bar(T_AB_bar, hpA_);
        Eigen::Vector4d hpB_bar = itpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
           (hpB_bar - hpB_) / eps;
        dhpB_dT_AB_numeric_.col(i) = ratio;
      }

      for (int i = 0; i < 4; ++i) {
        Eigen::Matrix<double, 4, 1> delta;
        delta.setZero();
        delta(i) = eps;
        Eigen::Vector4d hpA_bar = hpA_ + delta;
        okvis::InverseTransformPointJacobian itpj_bar(T_AB_, hpA_bar);
        Eigen::Vector4d hpB_bar = itpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
            (hpB_bar - hpB_) / eps;
        dhpB_dhpA_numeric_.col(i) = ratio;
      }
    }

    void check() const {
      Eigen::Matrix<double, 4, 4> dhpB_dhpA;
      itpj_.dhpB_dhpA(&dhpB_dhpA);
      EXPECT_LT((dhpB_dhpA_numeric_ - dhpB_dhpA).lpNorm<Eigen::Infinity>(), tol)
          << "dhpB_dhpA_numeric_\n"
          << dhpB_dhpA_numeric_ << "\ndhpB_dhpA\n"
          << dhpB_dhpA;

      Eigen::Matrix<double, 4, 6> dhpB_dT_AB;
      itpj_.dhpB_dT_AB(&dhpB_dT_AB);
      EXPECT_LT((dhpB_dT_AB_numeric_ - dhpB_dT_AB).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpB_dT_AB_numeric_\n"
          << dhpB_dT_AB_numeric_ << "\ndhpB_dT_AB\n"
          << dhpB_dT_AB;
    }

    void checkAgainstMultipleTransformJacobian() const {
      okvis::MultipleTransformPointJacobian mtpj({T_AB_}, {-1}, hpA_);
      Eigen::Matrix<double, 4, 4> dhpB_dhpA = mtpj.dp_dpoint();
      EXPECT_LT((dhpB_dhpA_numeric_ - dhpB_dhpA).lpNorm<Eigen::Infinity>(), tol)
          << "dhpB_dhpA_numeric_\n"
          << dhpB_dhpA_numeric_ << "\ndhpB_dhpA\n"
          << dhpB_dhpA;

      Eigen::Matrix<double, 4, 6> dhpB_dT_AB = mtpj.dp_dT(0u);
      EXPECT_LT((dhpB_dT_AB_numeric_ - dhpB_dT_AB).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpB_dT_AB_numeric_\n"
          << dhpB_dT_AB_numeric_ << "\ndhpB_dT_AB\n"
          << dhpB_dT_AB;
    }

    okvis::kinematics::Transformation T_AB_;
    Eigen::Vector4d hpA_;
    Eigen::Vector4d hpB_;
    okvis::InverseTransformPointJacobian itpj_;

    Eigen::Matrix<double, 4, 4> dhpB_dhpA_numeric_;
    Eigen::Matrix<double, 4, 6> dhpB_dT_AB_numeric_;
    const double eps = 1e-6;
    const double tol = 1e-6;
};

TEST_F(InverseTransformPointJacobianTest, dhp_dhp) {
  check();
}

TEST_F(InverseTransformPointJacobianTest, closeLoop) {
  okvis::TransformPointJacobian tpj(T_AB_, itpj_.evaluate());
  EXPECT_LT((tpj.evaluate() - hpA_).lpNorm<Eigen::Infinity>(), tol);
}

TEST_F(InverseTransformPointJacobianTest, MultipleTransform) {
  checkAgainstMultipleTransformJacobian();
}

class MultipleTransformPointJacobianTest : public ::testing::Test {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MultipleTransformPointJacobianTest() {}
    void SetUp() override {
      srand((unsigned int)time(0));  // comment this for deterministic behavior
      T_AB_.setRandom();
      T_CB_.setRandom();
      hpC_.setRandom();
      mtpj_.initialize({T_AB_, T_CB_}, {1, -1}, hpC_);
      hpA_ = mtpj_.evaluate();
      computeNumericJacobians();
    }

    void computeNumericJacobians() {
      for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta;
        delta.setZero();
        delta(i) = eps;
        okvis::kinematics::Transformation T_AB_bar = T_AB_;
        T_AB_bar.oplus(delta);
        okvis::MultipleTransformPointJacobian mtpj_bar({T_AB_bar, T_CB_}, {1, -1}, hpC_);
        Eigen::Vector4d hpA_bar = mtpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
           (hpA_bar - hpA_) / eps;
        dhpA_dT_AB_numeric_.col(i) = ratio;
      }

      for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta;
        delta.setZero();
        delta(i) = eps;
        okvis::kinematics::Transformation T_CB_bar = T_CB_;
        T_CB_bar.oplus(delta);
        okvis::MultipleTransformPointJacobian mtpj_bar({T_AB_, T_CB_bar}, {1, -1}, hpC_);
        Eigen::Vector4d hpA_bar = mtpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
           (hpA_bar - hpA_) / eps;
        dhpA_dT_CB_numeric_.col(i) = ratio;
      }

      for (int i = 0; i < 4; ++i) {
        Eigen::Matrix<double, 4, 1> delta;
        delta.setZero();
        delta(i) = eps;
        Eigen::Vector4d hpC_bar = hpC_ + delta;
        okvis::MultipleTransformPointJacobian mtpj_bar({T_AB_, T_CB_}, {1, -1}, hpC_bar);
        Eigen::Vector4d hpA_bar = mtpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
            (hpA_bar - hpA_) / eps;
        dhpA_dhpC_numeric_.col(i) = ratio;
      }
    }

    void check() const {
      Eigen::Matrix<double, 4, 4> dhpA_dhpC = mtpj_.dp_dpoint();
      EXPECT_LT((dhpA_dhpC_numeric_ - dhpA_dhpC).lpNorm<Eigen::Infinity>(), tol)
          << "dhpA_dhpC_numeric_\n"
          << dhpA_dhpC_numeric_ << "\ndhpA_dhpC\n"
          << dhpA_dhpC;

      Eigen::Matrix<double, 4, 6> dhpA_dT_AB = mtpj_.dp_dT(0u);
      EXPECT_LT((dhpA_dT_AB_numeric_ - dhpA_dT_AB).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpA_dT_AB_numeric_\n"
          << dhpA_dT_AB_numeric_ << "\ndhpA_dT_AB\n"
          << dhpA_dT_AB;

      Eigen::Matrix<double, 4, 6> dhpA_dT_CB = mtpj_.dp_dT(1u);
      EXPECT_LT((dhpA_dT_CB_numeric_ - dhpA_dT_CB).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpA_dT_CB_numeric_\n"
          << dhpA_dT_CB_numeric_ << "\ndhpA_dT_CB\n"
          << dhpA_dT_CB;
    }

    okvis::kinematics::Transformation T_AB_;
    okvis::kinematics::Transformation T_CB_;
    Eigen::Vector4d hpC_;
    Eigen::Vector4d hpA_;
    okvis::MultipleTransformPointJacobian mtpj_;

    Eigen::Matrix<double, 4, 4> dhpA_dhpC_numeric_;
    Eigen::Matrix<double, 4, 6> dhpA_dT_AB_numeric_;
    Eigen::Matrix<double, 4, 6> dhpA_dT_CB_numeric_;
    const double eps = 1e-6;
    const double tol = 1e-6;
};

TEST_F(MultipleTransformPointJacobianTest, ABinvxPoint) {
  check();
}


class MultipleTransformPointJacobianTest2 : public ::testing::Test {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    MultipleTransformPointJacobianTest2() {}
    void SetUp() override {
      srand((unsigned int)time(0));  // comment this for deterministic behavior
      T_BA_.setRandom();
      T_BC_.setRandom();
      hpC_.setRandom();
      mtpj_.initialize({T_BA_, T_BC_}, {-1, 1}, hpC_);
      hpA_ = mtpj_.evaluate();
      computeNumericJacobians();
    }

    void computeNumericJacobians() {
      for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta;
        delta.setZero();
        delta(i) = eps;
        okvis::kinematics::Transformation T_BA_bar = T_BA_;
        T_BA_bar.oplus(delta);
        okvis::MultipleTransformPointJacobian mtpj_bar({T_BA_bar, T_BC_}, {-1, 1}, hpC_);
        Eigen::Vector4d hpA_bar = mtpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
           (hpA_bar - hpA_) / eps;
        dhpA_dT_BA_numeric_.col(i) = ratio;
      }

      for (int i = 0; i < 6; ++i) {
        Eigen::Matrix<double, 6, 1> delta;
        delta.setZero();
        delta(i) = eps;
        okvis::kinematics::Transformation T_BC_bar = T_BC_;
        T_BC_bar.oplus(delta);
        okvis::MultipleTransformPointJacobian mtpj_bar({T_BA_, T_BC_bar}, {-1, 1}, hpC_);
        Eigen::Vector4d hpA_bar = mtpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
           (hpA_bar - hpA_) / eps;
        dhpA_dT_BC_numeric_.col(i) = ratio;
      }

      for (int i = 0; i < 4; ++i) {
        Eigen::Matrix<double, 4, 1> delta;
        delta.setZero();
        delta(i) = eps;
        Eigen::Vector4d hpC_bar = hpC_ + delta;
        okvis::MultipleTransformPointJacobian mtpj_bar({T_BA_, T_BC_}, {-1, 1}, hpC_bar);
        Eigen::Vector4d hpA_bar = mtpj_bar.evaluate();
        Eigen::Matrix<double, 4, 1> ratio =
            (hpA_bar - hpA_) / eps;
        dhpA_dhpC_numeric_.col(i) = ratio;
      }
    }

    void check() const {
      Eigen::Matrix<double, 4, 4> dhpA_dhpC = mtpj_.dp_dpoint();
      EXPECT_LT((dhpA_dhpC_numeric_ - dhpA_dhpC).lpNorm<Eigen::Infinity>(), tol)
          << "dhpA_dhpC_numeric_\n"
          << dhpA_dhpC_numeric_ << "\ndhpA_dhpC\n"
          << dhpA_dhpC;

      Eigen::Matrix<double, 4, 6> dhpA_dT_BA = mtpj_.dp_dT(0u);
      EXPECT_LT((dhpA_dT_BA_numeric_ - dhpA_dT_BA).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpA_dT_BA_numeric_\n"
          << dhpA_dT_BA_numeric_ << "\ndhpA_dT_BA\n"
          << dhpA_dT_BA;

      Eigen::Matrix<double, 4, 6> dhpA_dT_BC = mtpj_.dp_dT(1u);
      EXPECT_LT((dhpA_dT_BC_numeric_ - dhpA_dT_BC).lpNorm<Eigen::Infinity>(),
                tol)
          << "dhpA_dT_BC_numeric_\n"
          << dhpA_dT_BC_numeric_ << "\ndhpA_dT_BC\n"
          << dhpA_dT_BC;
    }

    okvis::kinematics::Transformation T_BA_;
    okvis::kinematics::Transformation T_BC_;
    Eigen::Vector4d hpC_;
    Eigen::Vector4d hpA_;
    okvis::MultipleTransformPointJacobian mtpj_;

    Eigen::Matrix<double, 4, 4> dhpA_dhpC_numeric_;
    Eigen::Matrix<double, 4, 6> dhpA_dT_BA_numeric_;
    Eigen::Matrix<double, 4, 6> dhpA_dT_BC_numeric_;
    const double eps = 1e-6;
    const double tol = 1e-6;
};

TEST_F(MultipleTransformPointJacobianTest2, AinvBxPoint) {
  check();
}
