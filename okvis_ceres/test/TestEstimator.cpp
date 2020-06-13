/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 * 
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Dec 30, 2014
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

#include <gtest/gtest.h>
#include <okvis/Estimator.hpp>
#include <okvis/IdProvider.hpp>
#include <okvis/MultiFrame.hpp>
#include <okvis/cameras/PinholeCamera.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/cameras/EquidistantDistortion.hpp>
#include <okvis/ceres/PoseParameterBlock.hpp>
#include <okvis/ceres/SpeedAndBiasParameterBlock.hpp>
#include <okvis/ceres/HomogeneousPointParameterBlock.hpp>
#include <okvis/ceres/ImuError.hpp>
#include <okvis/ceres/ReprojectionError.hpp>
#include <okvis/ceres/PoseError.hpp>
#include <okvis/ceres/SpeedAndBiasError.hpp>
#include <okvis/ceres/RelativePoseError.hpp>
#include <okvis/assert_macros.hpp>
#include <msckf/PointLandmarkSimulation.hpp>


TEST(okvisTestSuite, Estimator) {
  //srand((unsigned int) time(0)); // disabled: make unit tests deterministic...

  // different cases of camera extrinsics;
  for (size_t c = 0; c < 4; ++c) {

    OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error);

    const double DURATION = 10.0;  // 10 seconds motion
    const double IMU_RATE = 100.0;  // 1 kHz
    const double DT = 1.0 / IMU_RATE;  // time increments

    // set the imu parameters
    okvis::ImuParameters imuParameters;
    imuParameters.a0.setZero();
    imuParameters.g = 9.81;
    imuParameters.a_max = 1000.0;
    imuParameters.g_max = 1000.0;
    imuParameters.rate = 1000;  // 1 kHz
    imuParameters.sigma_g_c = 6.0e-4;
    imuParameters.sigma_a_c = 2.0e-3;
    imuParameters.sigma_gw_c = 3.0e-6;
    imuParameters.sigma_aw_c = 2.0e-5;
    imuParameters.tau = 3600.0;
    std::cout << "case " << c % 2 << ", " << c / 2 << std::endl;

    // let's generate a really stupid motion: constant translation
    okvis::SpeedAndBias speedAndBias;
    speedAndBias.setZero();
    speedAndBias.head<3>() = Eigen::Vector3d(0, 1, 0);
    okvis::ImuMeasurementDeque imuMeasurements;
    okvis::ImuSensorReadings nominalImuSensorReadings(
        Eigen::Vector3d::Zero(), Eigen::Vector3d(0, 0, imuParameters.g));
    okvis::Time t0 = okvis::Time::now();
    for (size_t i = 0; i <= DURATION * IMU_RATE; ++i) {
      Eigen::Vector3d gyr = nominalImuSensorReadings.gyroscopes
          + Eigen::Vector3d::Random() * imuParameters.sigma_g_c * sqrt(DT);
      Eigen::Vector3d acc = nominalImuSensorReadings.accelerometers
          + Eigen::Vector3d::Random() * imuParameters.sigma_a_c * sqrt(DT);
      imuMeasurements.push_back(
          okvis::ImuMeasurement(t0 + okvis::Duration(DT * i),
                                okvis::ImuSensorReadings(gyr, acc)));
    }

    // create the map
    std::shared_ptr<okvis::ceres::Map> mapPtr(new okvis::ceres::Map);

    // camera extrinsics:
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_0(
        new okvis::kinematics::Transformation());
    std::shared_ptr<const okvis::kinematics::Transformation> T_SC_1(
        new okvis::kinematics::Transformation(Eigen::Vector3d(0,0.1,0),Eigen::Quaterniond(1,0,0,0)));

    // some parameters on how to do the online estimation:
    okvis::ExtrinsicsEstimationParameters extrinsicsEstimationParameters;
    extrinsicsEstimationParameters.sigma_absolute_translation = 1.0e-3
        * (c % 2);
    extrinsicsEstimationParameters.sigma_absolute_orientation = 1.0e-4
        * (c % 2);
    extrinsicsEstimationParameters.sigma_c_relative_translation = 1e-8
        * (c / 2);
    extrinsicsEstimationParameters.sigma_c_relative_orientation = 1e-7
        * (c / 2);

    // set up camera with intrinsics
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry0(
        okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());
    std::shared_ptr<const okvis::cameras::CameraBase> cameraGeometry1(
        okvis::cameras::PinholeCamera<okvis::cameras::EquidistantDistortion>::createTestObject());

    // create an N-camera system:
    std::shared_ptr<okvis::cameras::NCameraSystem> cameraSystem(
        new okvis::cameras::NCameraSystem);
    cameraSystem->addCamera(T_SC_0, cameraGeometry0,
                            okvis::cameras::NCameraSystem::DistortionType::Equidistant);
    cameraSystem->addCamera(T_SC_1, cameraGeometry1,
                            okvis::cameras::NCameraSystem::DistortionType::Equidistant);

    // create an Estimator
    okvis::Estimator estimator(mapPtr);

    // create landmark grid
    const okvis::kinematics::Transformation T_WS_0;
    std::vector<Eigen::Vector4d,
        Eigen::aligned_allocator<Eigen::Vector4d> > homogeneousPoints;
    std::vector<uint64_t> lmIds;
    for (double y = -10.0; y <= DURATION * T_SC_1->r()[1] + 10.0; y += 0.5) {
      for (double z = -10.0; z <= 10.0; z += 0.5) {
        homogeneousPoints.push_back(Eigen::Vector4d(3.0, y, z, 1));
        lmIds.push_back(okvis::IdProvider::instance().newId());
        estimator.addLandmark(lmIds.back(), homogeneousPoints.back());
      }
    }

    // add sensors
    estimator.addCameraParameterStds(extrinsicsEstimationParameters);
    estimator.addCameraParameterStds(extrinsicsEstimationParameters);
    estimator.addImu(imuParameters);
    std::shared_ptr<okvis::MultiFrame> prevkeymf;

    const size_t K = 6;
    uint64_t id = -1;
    okvis::kinematics::Transformation T_WS_est;
    okvis::SpeedAndBias speedAndBias_est;
    for (size_t k = 0; k < K + 1; ++k) {
      // calculate the transformation
      okvis::kinematics::Transformation T_WS(
          T_WS_0.r() + speedAndBias.head<3>() * double(k) * DURATION / double(K),
          T_WS_0.q());

      // assemble a multi-frame
      std::shared_ptr<okvis::MultiFrame> mf(new okvis::MultiFrame);
      mf->setId(okvis::IdProvider::instance().newId());
      okvis::Time nframeStamp = t0 + okvis::Duration(double(k) * DURATION / double(K));
      mf->setTimestamp(nframeStamp);

      // reference ID will be and stay the first frame added.
      id = mf->id();

      // add frames
      mf->resetCameraSystemAndFrames(*cameraSystem);
      for (size_t c = 0; c < cameraSystem->numCameras(); ++c) {
        mf->setTimestamp(c, nframeStamp);
      }
      // add it in the window to create a new time instance
      estimator.addStates(mf, imuMeasurements, k % 3 == 0);
      std::cout << "Frame " << k << " successfully added." << std::endl;

      estimator.get_T_WS(mf->id(), T_WS_est);

      // now let's add also landmark observations
      std::vector<std::vector<size_t>> lmkIndices; // for every keypoint
      std::vector<std::vector<int>> keypointIndices; // for every landmark
      double imageNoiseStd = 0.7;
      PointLandmarkSimulation::projectLandmarksToNFrame(
          homogeneousPoints, T_WS, cameraSystem, mf, &lmkIndices,
          &keypointIndices, &imageNoiseStd);

      for (size_t i = 0; i < lmkIndices.size(); ++i) {
        for (size_t j = 0; j < lmkIndices[i].size(); ++j) {
          size_t lmkIndex = lmkIndices[i][j];
          size_t kpIndex = j;
          mf->setLandmarkId(i, j, lmIds[lmkIndex]);
          estimator.addObservation<okvis::cameras::PinholeCamera<
              okvis::cameras::EquidistantDistortion>>(lmIds[lmkIndex], mf->id(),
                                                      i, kpIndex);
        }
      }
      // run the optimization
      estimator.optimize(10, 4, false);

      std::shared_ptr<okvis::LoopQueryKeyframeMessage> queryKeyframe;
      estimator.getLoopQueryKeyframeMessage(mf, &queryKeyframe);
      bool isKf = (k % 3 == 0);
      // check number of landmarks, landmark positions are in camera frame,
      // the constraint list, poses, etc.
      if (isKf) {
        EXPECT_TRUE(queryKeyframe != nullptr);
        EXPECT_EQ(queryKeyframe->id_, mf->id());
        EXPECT_EQ(queryKeyframe->stamp_, mf->timestamp());
        okvis::kinematics::Transformation T_WS_kf;
        estimator.get_T_WS(mf->id(), T_WS_kf);
        EXPECT_LT((queryKeyframe->T_WB_.coeffs() - T_WS_kf.coeffs()).lpNorm<Eigen::Infinity>(), 1e-7);
        EXPECT_LT(queryKeyframe->getCovariance().lpNorm<Eigen::Infinity>(), 1e-7);

        if (k == 0) {
          EXPECT_EQ(queryKeyframe->odometryConstraintList().size(), 0u);
        } else {
          okvis::kinematics::Transformation T_WS_prevkf;
          estimator.get_T_WS(prevkeymf->id(), T_WS_prevkf);
          EXPECT_GE(queryKeyframe->odometryConstraintList().size(), 1u);
          std::shared_ptr<const okvis::NeighborConstraintMessage>
              constraintMessage =
                  queryKeyframe->odometryConstraintList().at(0u);
          EXPECT_EQ(constraintMessage->core_.id_, prevkeymf->id());
          EXPECT_EQ(constraintMessage->core_.stamp_, prevkeymf->timestamp());
          EXPECT_LT((constraintMessage->core_.T_BBr_.coeffs() -
                     (T_WS_prevkf.inverse() * T_WS_kf).coeffs())
                        .lpNorm<Eigen::Infinity>(),
                    1e-7);
          EXPECT_EQ(constraintMessage->core_.type_,
                    okvis::PoseConstraintType::Odometry);
          EXPECT_LT((constraintMessage->core_.squareRootInfo_ -
                     Eigen::Matrix<double, 6, 6>::Identity())
                        .lpNorm<Eigen::Infinity>(),
                    1e-7);
          EXPECT_LT((constraintMessage->T_WB_.coeffs() - T_WS_prevkf.coeffs())
                        .lpNorm<Eigen::Infinity>(),
                    1e-7);
        }
        prevkeymf = mf;
        const std::vector<int>& kpIndices = queryKeyframe->keypointIndexForLandmarkList();

        const std::vector<Eigen::Vector4d,
                          Eigen::aligned_allocator<Eigen::Vector4d>>&
            lmkPositions = queryKeyframe->landmarkPositionList();
        EXPECT_EQ(kpIndices.size(), lmkPositions.size());

        for (size_t k = 0; k < lmkPositions.size(); ++k) {
          int kpIndexForLmk = kpIndices[k];
          int lmkIndex = lmkIndices[0][kpIndexForLmk];
          Eigen::Vector4d hpW = homogeneousPoints[lmkIndex];
          okvis::MapPoint mp;
          estimator.getLandmark(lmIds[lmkIndex], mp);
          hpW = mp.pointHomog;
          Eigen::Vector4d hpB = T_WS_kf.inverse() * hpW;
          EXPECT_LT((lmkPositions[k] - hpB).lpNorm<Eigen::Infinity>(), 1e-7);
        }
      } else {
        EXPECT_FALSE(queryKeyframe);
      }
    }
    std::cout << "== TRY MARGINALIZATION ==" << std::endl;
    // try out the marginalization strategy
    okvis::MapPointVector removedLandmarks;
    estimator.applyMarginalizationStrategy(2, 3, removedLandmarks);
    // run the optimization
    std::cout << "== LAST OPTIMIZATION ==" << std::endl;
    estimator.optimize(10, 4, false);

    // get the estimates
    estimator.get_T_WS(id, T_WS_est);
    estimator.getSpeedAndBias(id, 0, speedAndBias_est);

    // inspect convergence:
    okvis::kinematics::Transformation T_WS(
         T_WS_0.r() + speedAndBias.head<3>() * DURATION,
         T_WS_0.q());

    std::cout << "estimated T_WS: " << std::endl << T_WS_est.T() << std::endl;
    std::cout << "correct T_WS: " << std::endl << T_WS.T() << std::endl;

    std::cout << (speedAndBias_est - speedAndBias).norm() << std::endl;

    EXPECT_LT((speedAndBias_est - speedAndBias).norm(), 0.05)
        << "speed and biases not close enough";
    EXPECT_LT(2 * (T_WS.q() * T_WS_est.q().inverse()).vec().norm(), 1e-2)
        << "quaternions not close enough";
    EXPECT_LT((T_WS.r() - T_WS_est.r()).norm(), 5e-1)
        << "translation not close enough";
  }
}
