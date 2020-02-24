#include "okvis/FullStateWithExtrinsicsAsCallback.hpp"

namespace okvis {
void FullStateWithExtrinsicsAsCallback::save(
    const okvis::Time &t, const okvis::kinematics::Transformation &T_WS,
    const Eigen::Matrix<double, 9, 1> &speedAndBiases,
    const Eigen::Matrix<double, 3, 1> & /*omega_S*/,
    const int frameIdInSource,
    const std::vector<
        okvis::kinematics::Transformation,
        Eigen::aligned_allocator<okvis::kinematics::Transformation>>
        &extrinsics) {
  const std::string datafile_separator = ",";
  Eigen::Vector3d p_WS_W = T_WS.r();
  Eigen::Quaterniond q_WS = T_WS.q();
  std::stringstream time;
  time << t.sec << std::setw(9) << std::setfill('0') << t.nsec;
  output_stream_ << time.str() << datafile_separator << frameIdInSource
                 << datafile_separator << std::setprecision(6) << p_WS_W[0]
                 << datafile_separator << p_WS_W[1] << datafile_separator
                 << p_WS_W[2] << datafile_separator << q_WS.x()
                 << datafile_separator << q_WS.y() << datafile_separator
                 << q_WS.z() << datafile_separator << q_WS.w()
                 << datafile_separator << speedAndBiases[0]
                 << datafile_separator << speedAndBiases[1]
                 << datafile_separator << speedAndBiases[2]
                 << datafile_separator << speedAndBiases[3]
                 << datafile_separator << speedAndBiases[4]
                 << datafile_separator << speedAndBiases[5]
                 << datafile_separator << speedAndBiases[6]
                 << datafile_separator << speedAndBiases[7]
                 << datafile_separator << speedAndBiases[8];
  for (size_t i = 0; i < extrinsics.size(); ++i) {
    Eigen::Vector3d p_BCi = extrinsics[i].r();
    Eigen::Quaterniond q_BCi = extrinsics[i].q();
    output_stream_ << datafile_separator << p_BCi[0] << datafile_separator
                   << p_BCi[1] << datafile_separator << p_BCi[2]
                   << datafile_separator << q_BCi.x() << datafile_separator
                   << q_BCi.y() << datafile_separator << q_BCi.z()
                   << datafile_separator << q_BCi.w();
  }
  output_stream_ << std::endl;
}
FullStateWithExtrinsicsAsCallback::FullStateWithExtrinsicsAsCallback(const std::string &output_file)
    : output_file_(output_file), output_stream_(output_file) {
  if (!output_stream_.is_open()) {
    std::cerr << "Warn: unable to open " << output_file << std::endl;
  }
}
FullStateWithExtrinsicsAsCallback::~FullStateWithExtrinsicsAsCallback() { output_stream_.close(); }
} // namespace okvis
