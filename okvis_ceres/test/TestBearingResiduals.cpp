#include <gtest/gtest.h>

#include <msckf/BearingResiduals.hpp>

TEST(BearingResiduals, orderedIndicesMAJ1) {
  int totalObs = 7;
  std::vector<int> ma{0, 1};
  std::vector<int> orderedIndicesExpected{0, 1, 2, 3, 4, 5, 6};
  std::vector<int> result;
  msckf::orderedIndicesMAJ(ma, totalObs, &result);

  for (size_t j = 0u; j < result.size(); ++j) {
      EXPECT_EQ(result[j], orderedIndicesExpected[j]);
  }
}

TEST(BearingResiduals, orderedIndicesMAJ2) {
  int totalObs = 7;
  std::vector<int> ma{1, 0};
  std::vector<int> orderedIndicesExpected{1, 0, 2, 3, 4, 5, 6};
  std::vector<int> result;
  msckf::orderedIndicesMAJ(ma, totalObs, &result);
  for (size_t j = 0u; j < result.size(); ++j) {
      EXPECT_EQ(result[j], orderedIndicesExpected[j]);
  }
}

TEST(BearingResiduals, orderedIndicesMAJ3) {
  int totalObs = 7;
  std::vector<int> ma{1, 2};
  std::vector<int> orderedIndicesExpected{1, 2, 0, 3, 4, 5, 6};
  std::vector<int> result;
  msckf::orderedIndicesMAJ(ma, totalObs, &result);

  for (size_t j = 0u; j < result.size(); ++j) {
      EXPECT_EQ(result[j], orderedIndicesExpected[j]);
  }
}

TEST(BearingResiduals, orderedIndicesMAJ4) {
  int totalObs = 7;
  std::vector<int> ma{0, 2};
  std::vector<int> orderedIndicesExpected{0, 2, 1, 3, 4, 5, 6};
  std::vector<int> result;
  msckf::orderedIndicesMAJ(ma, totalObs, &result);
  for (size_t j = 0u; j < result.size(); ++j) {
      EXPECT_EQ(result[j], orderedIndicesExpected[j]);
  }
}

TEST(BearingResiduals, orderedIndicesMAJ5) {
  int totalObs = 7;
  std::vector<int> ma{2, 5};
  std::vector<int> orderedIndicesExpected{2, 5, 0, 3, 4, 1, 6};
  std::vector<int> result;
  msckf::orderedIndicesMAJ(ma, totalObs, &result);
  for (size_t j = 0u; j < result.size(); ++j) {
      EXPECT_EQ(result[j], orderedIndicesExpected[j]);
  }
}
