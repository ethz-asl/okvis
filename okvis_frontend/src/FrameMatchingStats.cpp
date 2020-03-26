#include "msckf/FrameMatchingStats.hpp"
namespace opengv {
void findMatches(const okvis::Estimator& estimator,
                 std::shared_ptr<okvis::MultiFrame> frameAPtr, size_t camIdA,
                 std::shared_ptr<okvis::MultiFrame> frameBPtr, size_t camIdB,
                 okvis::Matches* matches) {
  size_t numKeypointsA = frameAPtr->numKeypoints(camIdA);
  size_t numKeypointsB = frameBPtr->numKeypoints(camIdB);
  matches->reserve(std::min(numKeypointsA, numKeypointsB));
  std::map<uint64_t, size_t> idMap;
  for (size_t k = 0; k < numKeypointsB; ++k) {

    // get landmark id, if set
    uint64_t lmId = frameBPtr->landmarkId(camIdB, k);
    if (lmId == 0)
      continue;

    // check, if existing
    if (!estimator.isLandmarkAdded(lmId))
      continue;

    // remember it
    idMap.insert(std::pair<uint64_t, size_t>(lmId, k));
  }

  for (size_t k = 0; k < numKeypointsA; ++k) {
    // get landmark id, if set
    uint64_t lmId = frameAPtr->landmarkId(camIdA, k);
    if (lmId == 0)
      continue;

    std::map<uint64_t, size_t>::const_iterator it = idMap.find(lmId);
    if (it != idMap.end()) {
      // whohoo, let's insert it.
      matches->push_back(okvis::Match(k, it->second, 0.0));
    }
  }
}

void computeMatchStats(
    const std::vector<okvis::Match>& matches,
    std::shared_ptr<okvis::MultiFrame> frameBPtr, size_t camIdx,
    double* overlap, double* matchingRatio) {
  *overlap = 0;
  *matchingRatio = 0;
  // get the hull of all keypoints in current frame
  std::vector<cv::Point2f> frameBPoints, frameBHull;
  std::vector<cv::Point2f> frameBMatches, frameBMatchesHull;

  const size_t numB = frameBPtr->numKeypoints(camIdx);
  frameBPoints.reserve(numB);
  frameBMatches.reserve(matches.size());

  for (size_t k = 0; k < numB; ++k) {
    Eigen::Vector2d keypoint;
    frameBPtr->getKeypoint(camIdx, k, keypoint);
    frameBPoints.push_back(cv::Point2f(keypoint[0], keypoint[1]));
  }
  for (size_t k = 0; k < matches.size(); ++k) {
    const size_t idx2 = matches[k].idxB;
    Eigen::Vector2d keypoint;
    frameBPtr->getKeypoint(camIdx, idx2, keypoint);
    frameBMatches.push_back(cv::Point2f(keypoint[0], keypoint[1]));
  }

  if (frameBPoints.size() < 3) return;
  cv::convexHull(frameBPoints, frameBHull);
  if (frameBMatches.size() < 3) return;
  cv::convexHull(frameBMatches, frameBMatchesHull);

  // areas
  double frameBArea = cv::contourArea(frameBHull);
  double frameBMatchesArea = cv::contourArea(frameBMatchesHull);

  // overlap area
  *overlap = frameBMatchesArea / frameBArea;
  // matching ratio inside overlap area: count
  int pointsInFrameBMatchesArea = 0;
  if (frameBMatchesHull.size() > 2) {
    for (size_t k = 0; k < frameBPoints.size(); ++k) {
      if (cv::pointPolygonTest(frameBMatchesHull, frameBPoints[k], false) > 0) {
        pointsInFrameBMatchesArea++;
      }
    }
  }
  *matchingRatio = static_cast<double>(frameBMatches.size()) /
                   static_cast<double>(pointsInFrameBMatchesArea);
}

} // namespace opengv
