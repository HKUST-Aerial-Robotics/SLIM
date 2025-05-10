#ifndef LINE_INFO_H
#define LINE_INFO_H

#include <Eigen/Core>

#include <memory>

#include "factor/math_utility.h"
#include "transform.h"

namespace SLIM {
class LineInfo {
 public:
  typedef std::shared_ptr<LineInfo> Ptr;

  inline LineInfo() : omega_(&parameters_[0]), scale_(&parameters_[2]) {}

  inline LineInfo(const Eigen::Vector4d& vec)
  : omega_(&parameters_[0]), scale_(&parameters_[2]) {
    parameters_ = vec;
  }

  inline LineInfo(const LineInfo& other)
  : omega_(&parameters_[0]), scale_(&parameters_[2]), parameters_(other.parameters_) {}

  inline LineInfo(const Eigen::Vector3d& random_point, const Eigen::Vector3d& direction)
  : omega_(&parameters_[0]), scale_(&parameters_[2]) {
    parameterize(random_point, direction);
  }

  inline void parameterize(const Eigen::Vector3d& random_point, const Eigen::Vector3d& direction) {
    Eigen::Matrix3d Rzv = g2R(direction), Rvz = Rzv.transpose();
    Eigen::Vector3d ypr = R2ypr(Rzv);
    omega_.x() = ypr(2);
    omega_.y() = ypr(1);

    Eigen::Vector3d nearest_point = random_point - direction * direction.transpose() * random_point;
    Eigen::Vector3d dv = Rzv * nearest_point;
    scale_.x() = dv.x();
    scale_.y() = dv.y();
  }

  inline void transform(const Transform& T) {
    Eigen::Vector3d const n = get_normal();
    Eigen::Vector3d const c = get_center();
    parameterize(T * c, T.dcm() * n);
  }

  inline Eigen::Matrix3d Rvz() const {
    double sinr = std::sin(omega_[0]), cosr = std::cos(omega_[0]), sinp = std::sin(omega_[1]), cosp = std::cos(omega_[1]);
    Eigen::Matrix3d Rr, Rp;
    Rr << 1,     0,    0, 
          0,  cosr, sinr, 
          0, -sinr, cosr;
    Rp << cosp,  0, -sinp,
             0,  1,     0,
          sinp,  0,  cosp;

    return Rr * Rp;
  }

  inline Eigen::Vector3d get_center() const {
    return Rvz() * (Eigen::Vector3d::UnitX() * scale_[0] + Eigen::Vector3d::UnitY() * scale_[1]);
  }

  inline Eigen::Vector3d get_normal() const {
    return Rvz() * Eigen::Vector3d::UnitZ();
  }

  inline Eigen::Vector3d distance(const Eigen::Vector3d& q) const {
    Eigen::Vector3d const n = get_normal();
    Eigen::Vector3d const c = get_center();
    return (Eigen::Matrix3d::Identity() - n * n.transpose()) * (q - c);
  }

  inline Eigen::Vector3d pedal(const Eigen::Vector3d& q) const {
    Eigen::Vector3d const n = get_normal();
    Eigen::Vector3d const c = get_center();
    return c + n * n.transpose() * (q - c);
  }

  inline Eigen::Matrix<double, 6, 4, Eigen::RowMajor> jacobian() const {

    Eigen::Matrix<double, 6, 4> jacobian_matrix;
    jacobian_matrix.setZero();
    Eigen::Matrix3d R = Rvz();
    Eigen::Vector3d const c = R * (Eigen::Vector3d::UnitX() * scale_[0] + Eigen::Vector3d::UnitY() * scale_[1]);
    Eigen::Vector3d const n = R * Eigen::Vector3d::UnitZ();
    double sinr = std::sin(omega_[0]), cosr = std::cos(omega_[0]);
    double sinp = std::sin(omega_[1]), cosp = std::cos(omega_[1]);
    const double alpha = scale_[0], beta = scale_[1];

    Eigen::Vector3d scale_w = Eigen::Vector3d::UnitX() * scale_[0] + Eigen::Vector3d::UnitY() * scale_[1];

    Eigen::Matrix<double, 3, 2> dn_dw;
    dn_dw.col(0) <<     0,  cosr * cosp, -sinr * cosp;
    dn_dw.col(1) << -cosp, -sinr * sinp, -cosr * sinp;

    Eigen::Matrix<double, 3, 2> dc_dw;
    dc_dw.col(0) <<             0, cosr * sinp * alpha - sinr * beta, -sinr * sinp * alpha - cosr * beta;
    dc_dw.col(1) << -sinp * alpha,               sinr * cosp * alpha,                cosr * cosp * alpha;

    Eigen::Matrix<double, 3, 2> dc_ds = R.leftCols<2>();

    jacobian_matrix.block<3, 2>(0, 0) = dn_dw;
    jacobian_matrix.block<3, 2>(3, 0) = dc_dw;
    jacobian_matrix.block<3, 2>(3, 2) = dc_ds;

    return jacobian_matrix;
  }

  inline Eigen::Matrix<double, 4, 1>& parameters() {
    return parameters_;
  }

  Eigen::Map<Eigen::Vector2d> omega_;
  Eigen::Map<Eigen::Vector2d> scale_;
  Eigen::Vector4d parameters_;
};
}

#endif