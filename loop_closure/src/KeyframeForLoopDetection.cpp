#include <okvis/KeyframeForLoopDetection.hpp>
#include <okvis/kinematics/MatrixPseudoInverse.hpp>

namespace okvis {
NeighborConstraintInDatabase::NeighborConstraintInDatabase() {}

NeighborConstraintInDatabase::NeighborConstraintInDatabase(
    uint64_t id, okvis::Time stamp,
    const okvis::kinematics::Transformation& T_BBr,
    PoseConstraintType type) :
  id_(id), stamp_(stamp), T_BBr_(T_BBr), type_(type) {

}

NeighborConstraintInDatabase::~NeighborConstraintInDatabase() {}

NeighborConstraintMessage::NeighborConstraintMessage() {}

NeighborConstraintMessage::NeighborConstraintMessage(
    uint64_t id, okvis::Time stamp,
    const okvis::kinematics::Transformation& T_BnBr,
    const okvis::kinematics::Transformation& T_WB,
    PoseConstraintType type) :
  core_(id, stamp, T_BnBr, type), T_WB_(T_WB) {

}

NeighborConstraintMessage::~NeighborConstraintMessage() {}

void NeighborConstraintMessage::computeRelativePoseCovariance(
    const okvis::kinematics::Transformation& T_WBr,
    const Eigen::Matrix<double, 6, 6>& cov_T_WBr,
    Eigen::Matrix<double, 6, 6>* cov_T_BnBr) {
  // use T_WBr, T_WBn, cov_T_WBr_WB_, cov_T_WBn, cov_T_WBr to compute
  // cov_T_BnBr
  InverseTransformMultiplyJacobian itmj(T_WB_, T_WBr);
  Eigen::Matrix<double, 6, 6> Jzr, Jzn;
  itmj.dT_dT_WA(&Jzn);
  itmj.dT_dT_WB(&Jzr);
  Eigen::Matrix<double, 6, 6> crossTerm =
      Jzr * cov_T_WBr_T_WB_ * Jzn.transpose();
  *cov_T_BnBr = Jzr * cov_T_WBr * Jzr.transpose() + crossTerm.transpose() +
               crossTerm + Jzn * cov_T_WB_ * Jzn.transpose();
}

LoopQueryKeyframeMessage::LoopQueryKeyframeMessage() {}

LoopQueryKeyframeMessage::LoopQueryKeyframeMessage(uint64_t id, okvis::Time stamp,
                         const okvis::kinematics::Transformation& T_WB,
                         std::shared_ptr<const okvis::MultiFrame> multiframe)
    : id_(id), stamp_(stamp), T_WB_(T_WB), nframe_(multiframe) {}

LoopQueryKeyframeMessage::~LoopQueryKeyframeMessage() {}

KeyframeInDatabase::KeyframeInDatabase() {}

KeyframeInDatabase::KeyframeInDatabase(
    size_t dbowId, uint64_t vioId, okvis::Time stamp,
    const okvis::kinematics::Transformation& vio_T_WB,
    const Eigen::Matrix<double, 6, 6>& cov_T_WB)
    : dbowId_(dbowId), id_(vioId), stamp_(stamp),
      vio_T_WB_(vio_T_WB), cov_vio_T_WB_(cov_T_WB) {

}

void KeyframeInDatabase::setSquareRootInfoFromCovariance(
    size_t j, const Eigen::Matrix<double, 6, 6>& covRawError) {
  Eigen::Matrix<double, 6, 6> pseudoL;
  okvis::MatrixPseudoInverse::pseudoInverseSymmSqrt(
      covRawError, pseudoL,
      std::numeric_limits<double>::epsilon());
  constraintList_.at(j)->squareRootInfo_.noalias() = pseudoL.transpose();
}

}  // namespace okvis
