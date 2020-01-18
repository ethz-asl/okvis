#ifndef INCLUDE_MSCKF_POINT_LANDMARK_HPP_
#define INCLUDE_MSCKF_POINT_LANDMARK_HPP_

#include <vector>
#include <okvis/FrameTypedefs.hpp>

namespace msckf {
class PointLandmark {
public:
 PointLandmark(int modelId);

 // initialize a PointLandmark depending on the paramterization, refer to
 // triangulateALandmark
 std::vector<double> initializePointLandmark(const okvis::MapPoint& mp);

private:
  int modelId_;
  std::vector<double> parameters_;
};
}
#endif // INCLUDE_MSCKF_POINT_LANDMARK_HPP_
