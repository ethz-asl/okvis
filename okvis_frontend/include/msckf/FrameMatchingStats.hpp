#ifndef INCLUDE_MSCKF_FRAME_MATCHING_STATS_HPP_
#define INCLUDE_MSCKF_FRAME_MATCHING_STATS_HPP_

#include <okvis/Estimator.hpp>
#include <okvis/cameras/NCameraSystem.hpp>

namespace opengv {
/**
 * @brief find common landmarks between multiframeA camIdA and multiframeB camIdB.
 * @param estimator
 * @param frameAPtr
 * @param camIdA
 * @param frameBPtr
 * @param camIdB
 * @param matches[out]
 */
void findMatches(const okvis::Estimator& estimator,
                 std::shared_ptr<okvis::MultiFrame> frameAPtr, size_t camIdA,
                 std::shared_ptr<okvis::MultiFrame> frameBPtr, size_t camIdB,
                 okvis::Matches* matches);
/**
 * @brief compute overlap and matching ratio  between multiframeA camIdA and
 *     multiframeB camIdB. Info of multiframeA camIdA is embedded in matches.
 * @param matches
 * @param frameBPtr
 * @param camIdx
 * @param overlap[out]
 * @param matchingRatio[out]
 */
void computeMatchStats(
    const std::vector<okvis::Match>& matches,
    std::shared_ptr<okvis::MultiFrame> multiframeBPtr, size_t camIdx,
    double* overlap, double* matchingRatio);

} // namespace opengv
#endif // INCLUDE_MSCKF_FRAME_MATCHING_STATS_HPP_
