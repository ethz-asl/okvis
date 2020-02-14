#include "msckf/ceres/tiny_solver.h"
#include "gtest/gtest.h"

bool EvaluateResidualsAndJacobians(const double* parameters,
                                   double* residuals,
                                   double* jacobian) {
  double x = parameters[0];
  double y = parameters[1];
  double z = parameters[2];

  residuals[0] = x + 2*y + 4*z;
  residuals[1] = y * z;

  if (jacobian) {
    jacobian[0 * 2 + 0] = 1;
    jacobian[0 * 2 + 1] = 0;

    jacobian[1 * 2 + 0] = 2;
    jacobian[1 * 2 + 1] = z;

    jacobian[2 * 2 + 0] = 4;
    jacobian[2 * 2 + 1] = y;
  }
  return true;
}

class ExampleResidualsDynamic {
 public:
  typedef double Scalar;
  enum {
    NUM_RESIDUALS = Eigen::Dynamic,
    NUM_PARAMETERS = 3,
    NUM_LOCAL_PARAMETERS = 3,
  };

  int NumResiduals() const {
    return 2;
  }

  bool operator()(const double* parameters,
                  double* residuals,
                  double* jacobian) const {
    return EvaluateResidualsAndJacobians(parameters, residuals, jacobian);
  }
};

TEST(TinySolver, DynamicResiduals) {
  Eigen::Vector3d x0(0.76026643, -30.01799744, 0.55192142);
  ExampleResidualsDynamic f;
  Eigen::Vector3d x = x0;
  Eigen::Vector2d residuals;
  f(x.data(), residuals.data(), NULL);
  EXPECT_GT(residuals.squaredNorm() / 2.0, 1e-10);

  msckf::ceres::TinySolver<ExampleResidualsDynamic> solver;
  solver.Solve(f, &x);
  EXPECT_NEAR(0.0, solver.summary.final_cost, 1e-10);
}
