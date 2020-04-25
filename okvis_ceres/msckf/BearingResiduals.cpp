#include "msckf/BearingResiduals.hpp"
namespace msckf {

void orderedIndicesMAJ(const std::vector<size_t>& anchorIndices,
                       int totalObservations, std::vector<int>* maj) {
  maj->clear();
  maj->reserve(totalObservations);
  for (int i = 0; i < totalObservations; ++i) {
    maj->push_back(i);
  }
  size_t main = anchorIndices[0];
  size_t associate = anchorIndices[1];
  if (associate == 0u) {
    if (main == 1u) {
      std::swap(maj->at(0), maj->at(1));
    } else {
      std::swap(maj->at(0), maj->at(1));
      std::swap(maj->at(0), maj->at(main));
    }
  } else {
    std::swap(maj->at(0), maj->at(main));
    std::swap(maj->at(1), maj->at(associate));
  }
}

bool BearingResiduals::operator()(const double* parameters, double* residuals,
                                  double* jacobian) const {
  Eigen::Map<Eigen::Matrix<double, -1, 1>> resVec(residuals, numResiduals_, 1);
  Eigen::Map<Eigen::Matrix<double, -1, 3>> jacColMajor(
      jacobian, numResiduals_, 3);

  // main anchor
  LWF::ParallaxAnglePoint pap;
  pap.set(parameters);
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> jMain;
  jMain.leftCols(2) = pap.n_.getM();
  jMain.col(2).setZero();
  Eigen::Vector3d rMain =
      pap.n_.getVec() - pointDataPtr_->unitBearingList[majIndices_[0]];
  correct(&rMain, &jMain);
  resVec.head<3>() = rMain;
  if (jacobian) {
    jacColMajor.topRows<3>() = jMain;
  }

  // associate anchor and rest frames
  for (int j = 1; j < numResidualBlocks_; ++j) {
    DirectionFromParallaxAngleJacobian dfpaj(
        pointDataPtr_->T_WC_list.at(majIndices_[0]),
        pointDataPtr_->T_WC_list.at(majIndices_[1]).r(),
        pointDataPtr_->T_WC_list.at(majIndices_[j]).r(), pap);
    VectorNormalizationJacobian xi(dfpaj.evaluate());
    Eigen::Matrix<double, 3, 2> jni;
    dfpaj.dN_dni(&jni);
    Eigen::Vector3d jthetai;
    dfpaj.dN_dthetai(&jthetai);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> jRegular;
    xi.dxi_dvec(&jRegular);
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> jpap;
    jpap << jni, jthetai;
    jRegular = jRegular * jpap;
    Eigen::Vector3d rRegular =
        xi.normalized() - pointDataPtr_->T_WC_list[majIndices_[j]].C() *
                              pointDataPtr_->unitBearingList[majIndices_[j]];
    correct(&rRegular, &jRegular);
    resVec.segment<3>(j * 3) = rRegular;
    if (jacobian) {
      jacColMajor.block<3, 3>(j * 3, 0) = jRegular;
    }
  }
  return true;
}

}  // namespace msckf
