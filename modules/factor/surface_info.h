#ifndef SURFACE_INFO_H
#define SURFACE_INFO_H

#include <Eigen/Core>

#include "factor/math_utility.h"
namespace SLIM {
class SurfaceInfo {
 public:
  typedef std::shared_ptr<SurfaceInfo> Ptr;
  inline SurfaceInfo() : omega_(&parameters_[0]), dist_(&parameters_[2]) {}

  inline SurfaceInfo(const Eigen::Vector3d& vec)
  : omega_(&parameters_[0]), dist_(&parameters_[2]) {
    omega_ = vec.head<2>();
    dist_ = vec.tail<1>();
  }

  inline SurfaceInfo(const Eigen::Vector4d& vec)
  : omega_(&parameters_[0]), dist_(&parameters_[2]) {
    parameterize(vec.head<3>(), vec(3));
  }

  inline SurfaceInfo(const SurfaceInfo& other)
  : parameters_(other.parameters_), omega_(&parameters_[0]), dist_(&parameters_[2]) {}

  inline SurfaceInfo(const Eigen::Vector3d& random_point, const Eigen::Vector3d& direction)
  : omega_(&parameters_[0]), dist_(&parameters_[2]) {
    parameterize(direction, -direction.transpose() * random_point);
  }

  inline void parameterize(const Eigen::Vector3d& direction, const double dist) {
    Eigen::Matrix3d Rzv = g2R(direction), Rvz = Rzv.transpose();
    Eigen::Vector3d ypr = R2ypr(Rzv);
    omega_.x() = ypr(2);
    omega_.y() = ypr(1);
    dist_(0) = dist;
  }

  inline void transform(const Transform& T) {
    Eigen::Vector3d n = get_normal();
    Eigen::Vector3d c = pedal(Eigen::Vector3d::Zero());
    Eigen::Vector3d nt = T.dcm() * n;
    Eigen::Vector3d ct = T * c;
    parameterize(nt, -nt.transpose() * ct);
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

  inline double get_dist() const {
    return dist_(0);
  }

  inline Eigen::Vector3d get_normal() const {
    return Rvz() * Eigen::Vector3d::UnitZ();
  }

  inline double distance(const Eigen::Vector3d& q) const {
    return get_normal().dot(q) + dist_(0);
  }

  inline Eigen::Vector3d pedal(const Eigen::Vector3d& q) const {
    auto n = get_normal();
    auto d = n.dot(q) + dist_(0);
    return q - n * d;
  }

  inline Eigen::Matrix<double, 3, 2> subspace() const {
    Eigen::Vector3d z = get_normal();
    Eigen::Vector3d x = Eigen::Vector3d::Random().normalized();
    Eigen::Vector3d y = z.cross(x);
    Eigen::Vector3d xn, yn, zn;
    zn = z;
    yn = y - (y.dot(zn)) / (zn.dot(zn)) * zn;
    xn = x - (x.dot(zn)) / (zn.dot(zn)) * zn - (x.dot(yn)) / (yn.dot(yn)) * yn;
    Eigen::Matrix<double, 3, 2> subspace;
    subspace.col(0) = xn.normalized();
    subspace.col(1) = yn.normalized();
    assert(z.dot(subspace.col(0)) < 1e-8);
    assert(z.dot(subspace.col(1)) < 1e-8);
    return subspace;
  }

  inline Eigen::Matrix<double, 4, 3, Eigen::RowMajor> jacobian() const {

    Eigen::Matrix<double, 4, 3> jacobian_matrix;
    jacobian_matrix.setZero();
    Eigen::Matrix3d R = Rvz();
    auto n = R * Eigen::Vector3d::UnitZ();
    double sinr = std::sin(omega_[0]), cosr = std::cos(omega_[0]);
    double sinp = std::sin(omega_[1]), cosp = std::cos(omega_[1]);
    double d = dist_(0);

    Eigen::Matrix<double, 3, 2> dn_dw;
    dn_dw.col(0) <<     0,  cosr * cosp, -sinr * cosp;
    dn_dw.col(1) << -cosp, -sinr * sinp, -cosr * sinp;

    jacobian_matrix.block<3, 2>(0, 0) = dn_dw;
    jacobian_matrix(3, 2) = 1.0;

    return jacobian_matrix;
  }

  inline Eigen::Matrix<double, 3, 1>& parameters() {
    return parameters_;
  }

  Eigen::Map<Eigen::Vector2d> omega_;
  Eigen::Map<Eigen::Matrix<double, 1, 1>> dist_;
  Eigen::Vector3d parameters_;
};
}

#endif