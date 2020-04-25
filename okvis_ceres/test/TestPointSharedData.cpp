#include <gtest/gtest.h>
#include <msckf/PointSharedData.hpp>

class PointSharedDataTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int modulo = 7;
    uint64_t frameId = rand() % modulo + 1;
    int total = 9;
    int selected = 4;

    frameIds.reserve(total);
    std::vector<int> indices;
    indices.reserve(total);
    for (int i = 0; i < total; ++i) {
      psd.addKeypointObservation(
          okvis::KeypointIdentifier(frameId, 0, 0),
          std::shared_ptr<const okvis::ceres::ParameterBlock>(),
          std::rand() % 13 / 13.0);
      frameIds.push_back(frameId);
      indices.push_back(i);
      noise2dList.push_back(frameId);
      noise2dList.push_back(frameId);
      frameId += (rand() % modulo + 1);
    }

    std::random_shuffle(indices.begin(), indices.end());
    std::sort(indices.begin(), indices.begin() + selected);

    selectedFrameIds.reserve(selected);
    for (int j = 0; j < selected; ++j) {
      selectedFrameIds.push_back(frameIds[indices[j]]);
    }
    std::sort(indices.begin() + selected, indices.end());

    restFrameIds.reserve(total - selected);
    for (int j = selected; j < total; ++j) {
      restFrameIds.push_back(frameIds[indices[j]]);
    }
  }

  msckf::PointSharedData psd;
  std::vector<uint64_t> frameIds;
  std::vector<uint64_t> selectedFrameIds;
  std::vector<uint64_t> restFrameIds;

  std::vector<double> noise2dList;
};

TEST_F(PointSharedDataTest, removeExtraObservations) {
  psd.removeExtraObservations(selectedFrameIds, &noise2dList);
  std::vector<std::pair<uint64_t, size_t>> frameIdentifiers = psd.frameIds();
  EXPECT_EQ(selectedFrameIds.size(), frameIdentifiers.size());
  for (size_t j = 0; j < selectedFrameIds.size(); ++j) {
    EXPECT_EQ(selectedFrameIds[j], frameIdentifiers[j].first);
    EXPECT_EQ((uint64_t)noise2dList[2 * j], selectedFrameIds[j]);
    EXPECT_EQ((uint64_t)noise2dList[2 * j + 1], selectedFrameIds[j]);
  }
}

TEST_F(PointSharedDataTest, removeExtraObservationsLegacy) {
  psd.removeExtraObservationsLegacy(selectedFrameIds, &noise2dList);
  std::vector<std::pair<uint64_t, size_t>> frameIdentifiers = psd.frameIds();
  EXPECT_EQ(selectedFrameIds.size(), frameIdentifiers.size());
  for (size_t j = 0; j < selectedFrameIds.size(); ++j) {
    EXPECT_EQ(selectedFrameIds[j], frameIdentifiers[j].first);
    EXPECT_EQ((uint64_t)noise2dList[2 * j], selectedFrameIds[j]);
    EXPECT_EQ((uint64_t)noise2dList[2 * j + 1], selectedFrameIds[j]);
  }
}
