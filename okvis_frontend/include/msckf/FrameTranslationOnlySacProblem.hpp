
/**
 * @file FrameTranslationOnlySacProblem.hpp
 * @brief Header file for the FrameTranslationOnlySacProblem class.
 * @author Jianzhu Huai
 */

#ifndef INCLUDE_OKVIS_OPENGV_FRAMETRANSLATIONONLYSACPROBLEM_HPP_
#define INCLUDE_OKVIS_OPENGV_FRAMETRANSLATIONONLYSACPROBLEM_HPP_

#include <okvis/assert_macros.hpp>
#include <opengv/types.hpp>
#include <opengv/triangulation/methods.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac_problems/relative_pose/TranslationOnlySacProblem.hpp>
#include <opengv/relative_pose/FrameRelativeAdapter.hpp>

/**
 * \brief Namespace for classes extending the OpenGV library.
 */
namespace opengv {
/**
 * \brief The namespace for the sample consensus problems.
 */
namespace sac_problems {
/**
 * \brief The namespace for the relative pose methods.
 */
namespace relative_pose {

/**
 * \brief Provides functions for fitting a relative-pose model to a set of
 *        bearing-vector to point correspondences, using different algorithms (only
 *        central case). Used in a sample-consenus paradigm for rejecting
 *        outlier correspondences.
 */
class FrameTranslationOnlySacProblem : public TranslationOnlySacProblem {
 public:
  OKVIS_DEFINE_EXCEPTION(Exception,std::runtime_error)

  typedef TranslationOnlySacProblem base_t;

  /** The type of adapter that is expected by the methods */
  using base_t::adapter_t;
  /** The model we are trying to fit (transformation) */
  using base_t::model_t;

  /**
   * \brief FrameTranslationOnlySacProblem
   * \param[in] adapter Visitor holding bearing vector correspondences etc.
   * \param[in] algorithm The algorithm we want to use.
   * @warning Only okvis::relative_pose::FrameRelativeAdapter supported.
   */
  FrameTranslationOnlySacProblem(adapter_t & adapter)
      : base_t(adapter),
        adapterDerived_(
            *static_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter),
        "only opengv::absolute_pose::FrameRelativeAdapter supported");
  }

  /**
   * \brief FrameTranslationOnlySacProblem
   * \param[in] adapter Visitor holding bearing vector correspondences etc.
   * \param[in] algorithm The algorithm we want to use.
   * \param[in] indices A vector of indices to be used from all available
   *                    correspondences.
   * @warning Only okvis::relative_pose::FrameRelativeAdapter supported.
   */
  FrameTranslationOnlySacProblem(adapter_t & adapter,
                                 const std::vector<int> & indices)
      : base_t(adapter, indices),
        adapterDerived_(
            *static_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter)) {
    OKVIS_ASSERT_TRUE(
        Exception,
        dynamic_cast<opengv::relative_pose::FrameRelativeAdapter*>(&_adapter),
        "only opengv::absolute_pose::FrameRelativeAdapter supported");
  }
  virtual ~FrameTranslationOnlySacProblem() {
  }

  /**
   * \brief Compute the distances of all samples whith respect to given model
   *        coefficients.
   * \param[in] model The coefficients of the model hypothesis.
   * \param[in] indices The indices of the samples of which we compute distances.
   * \param[out] scores The resulting distances of the selected samples. Low
   *                    distances mean a good fit.
   */
  virtual void getSelectedDistancesToModel(const model_t & model,
                                           const std::vector<int> & indices,
                                           std::vector<double> & scores) const {
    translation_t translation = model.col(3);
    rotation_t rotation = model.block<3, 3>(0, 0);
    adapterDerived_.sett12(translation);
    adapterDerived_.setR12(rotation);

    model_t inverseSolution;
    inverseSolution.block<3, 3>(0, 0) = rotation.transpose();
    inverseSolution.col(3) = -inverseSolution.block<3, 3>(0, 0) * translation;

    Eigen::Matrix<double, 4, 1> p_hom;
    p_hom[3] = 1.0;

    for (size_t i = 0; i < indices.size(); i++) {
      p_hom.block<3, 1>(0, 0) = opengv::triangulation::triangulate2(
          adapterDerived_, indices[i]);
      bearingVector_t reprojection1 = p_hom.block<3, 1>(0, 0);
      bearingVector_t reprojection2 = inverseSolution * p_hom;
      reprojection1 = reprojection1 / reprojection1.norm();
      reprojection2 = reprojection2 / reprojection2.norm();
      bearingVector_t f1 = adapterDerived_.getBearingVector1(indices[i]);
      bearingVector_t f2 = adapterDerived_.getBearingVector2(indices[i]);

      //compute the score
      point_t error1 = (reprojection1 - f1);
      point_t error2 = (reprojection2 - f2);
      double error_squared1 = error1.transpose() * error1;
      double error_squared2 = error2.transpose() * error2;
      // sigmaAngle is the hypotenuse of the right triangle with its two legs
      // being the first two diagonal entries of covariance of a bearing vector
      // observation. Concretely, sigma_c = 0.8 * 8 / 12;
      // sigmaAngle = sqrt(2) * sigma_c * sigma_c / (fx * fx);
      scores.push_back(
          error_squared1 * 0.5 / adapterDerived_.getSigmaAngle1(indices[i])
              + error_squared2 * 0.5
                  / adapterDerived_.getSigmaAngle2(indices[i]));
    }
  }

 protected:
  /// The adapter holding the bearing, correspondences etc.
  opengv::relative_pose::FrameRelativeAdapter & adapterDerived_;

};

}
}
}

#endif /* INCLUDE_OKVIS_OPENGV_FRAMETRANSLATIONONLYSACPROBLEM_HPP_ */
