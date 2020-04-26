#include "msckf/MultipleTransformPointJacobian.hpp"
#include "msckf/InverseTransformPointJacobian.hpp"
#include "msckf/TransformPointJacobian.hpp"

namespace okvis {
Eigen::Vector4d MultipleTransformPointJacobian::evaluate() const {
  int j = (int)exponentList_.size() - 1;
  Eigen::Vector4d p = point_;
  for (auto riter = transformList_.rbegin(); riter != transformList_.rend();
       ++riter) {
    if (exponentList_[j] == -1) {
      p = riter->inverse() * p;
    } else {
      p = *riter * p;
    }
    --j;
  }
  return p;
}

void MultipleTransformPointJacobian::computeJacobians() {
  int j = (int)exponentList_.size();
  transformJacobianList_.resize(j + 1);
  transformJacobianList_[j].point_ = point_;
  transformJacobianList_[j].dpoint_dHeadTransform_.setZero();

  --j;
  Eigen::Vector4d p = point_;
  Eigen::Matrix<double, 4, 6> dp_dHeadTransform;
  for (auto riter = transformList_.rbegin(); riter != transformList_.rend();
       ++riter) {
    if (exponentList_[j] == -1) {
      InverseTransformPointJacobian itpj(*riter, p);
      p = itpj.evaluate();
      itpj.dhpB_dT_AB(&dp_dHeadTransform);
    } else {
      TransformPointJacobian tpj(*riter, p);
      p = tpj.evaluate();
      tpj.dhpA_dT_AB(&dp_dHeadTransform);
    }
    transformJacobianList_[j].point_ = p;
    transformJacobianList_[j].dpoint_dHeadTransform_ = dp_dHeadTransform;
    --j;
  }

  transformJacobianList_[0].cumulativeLeftTransform_.setIdentity();
  okvis::kinematics::Transformation leftCumulativeT;
  j = 0;
  for (auto iter = transformList_.begin(); iter != transformList_.end();
       ++iter) {
    if (exponentList_[j] == -1) {
      leftCumulativeT = leftCumulativeT * iter->inverse();
    } else {
      leftCumulativeT = leftCumulativeT * (*iter);
    }
    transformJacobianList_[j+1].cumulativeLeftTransform_ = leftCumulativeT;
    ++j;
  }
}

}  // namespace okvis
