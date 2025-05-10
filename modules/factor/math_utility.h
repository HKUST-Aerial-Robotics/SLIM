#ifndef MATH_UTILITY_H
#define MATH_UTILITY_H

#include <Eigen/Core>
#include <vector>

namespace SLIM {

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q) {
  Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
  ans << typename Derived::Scalar(0), -q(2), q(1),
      q(2), typename Derived::Scalar(0), -q(0),
      -q(1), q(0), typename Derived::Scalar(0);
  return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q) {
  Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
  ans(0, 0) = q.w(), ans.template block<1, 3>(0, 1) = -q.vec().transpose();
  ans.template block<3, 1>(1, 0) = q.vec(), ans.template block<3, 3>(1, 1) = q.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(q.vec());
  return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p) {
  Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
  ans(0, 0) = p.w(), ans.template block<1, 3>(0, 1) = -p.vec().transpose();
  ans.template block<3, 1>(1, 0) = p.vec(), ans.template block<3, 3>(1, 1) = p.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(p.vec());
  return ans;
}

template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta) {
  typedef typename Derived::Scalar Scalar_t;
  Eigen::Quaternion<Scalar_t> dq;
  Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
  half_theta /= static_cast<Scalar_t>(2.0);
  dq.w() = static_cast<Scalar_t>(1.0);
  dq.x() = half_theta.x();
  dq.y() = half_theta.y();
  dq.z() = half_theta.z();
  return dq;
}

template <typename Scalar>
static Eigen::Matrix<Scalar, 2, 2> ExpSO2(const Scalar &theta) {
  Eigen::Matrix<Scalar, 2, 2> R;
  Scalar cosx = std::cos(theta), sinx = std::sin(theta);
  R << cosx, -sinx, sinx, cosx;
  return R;
}

template <typename Scalar>
static Scalar LogSO2(const Eigen::Matrix<Scalar, 2, 2> &R) {
  return std::atan2(R(1, 0), R(0, 0));
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ExpSO3(const Eigen::MatrixBase<Derived> &theta) {
  typedef typename Derived::Scalar Scalar_t;
  Scalar_t theta_norm;
  Scalar_t theta_sq = theta.squaredNorm();
  Scalar_t imag_factor;
  Scalar_t real_factor;
  if (theta_sq < 1e-10) {
    Scalar_t theta_po4 = theta_sq * theta_sq;
    imag_factor = Scalar_t(0.5) - Scalar_t(1.0 / 48.0) * theta_sq +
                  Scalar_t(1.0 / 3840.0) * theta_po4;
    real_factor = Scalar_t(1) - Scalar_t(1.0 / 8.0) * theta_sq +
                  Scalar_t(1.0 / 384.0) * theta_po4;
  } else {
    theta_norm = std::sqrt(theta_sq);
    Scalar_t half_theta = Scalar_t(0.5) * (theta_norm);
    Scalar_t sin_half_theta = std::sin(half_theta);
    imag_factor = sin_half_theta / (theta_norm);
    real_factor = std::cos(half_theta);
  }
  Eigen::Quaternion<Scalar_t> q(real_factor, imag_factor * theta.x(), imag_factor * theta.y(), imag_factor * theta.z());
  return q.toRotationMatrix();
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 1> LogSO3(const Eigen::MatrixBase<Derived> &R) {
  typedef typename Derived::Scalar Scalar_t;
  auto q = Eigen::Quaternion<Scalar_t>(R);
  Scalar_t squared_n = q.vec().squaredNorm();
  Scalar_t w = q.w();
  Scalar_t two_atan_nbyw_by_n;
  if (squared_n < 1e-10) {
    Scalar_t squared_w = w * w;
    two_atan_nbyw_by_n = Scalar_t(2) / w - Scalar_t(2.0 / 3.0) * (squared_n) / (w * squared_w);
  } else {
    Scalar_t n = std::sqrt(squared_n);
    Scalar_t atan_nbyw = (w < Scalar_t(0)) ? Scalar_t(std::atan2(-n, -w)) : Scalar_t(std::atan2(n, w));
    two_atan_nbyw_by_n = Scalar_t(2) * atan_nbyw / n;
  }
  return two_atan_nbyw_by_n * q.vec();
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> ypr2R(const Eigen::MatrixBase<Derived> &ypr) {
  typedef typename Derived::Scalar Scalar_t;
  Scalar_t y = ypr(0);
  Scalar_t p = ypr(1);
  Scalar_t r = ypr(2);
  Eigen::Matrix<Scalar_t, 3, 3> Rz;
  Rz << cos(y), -sin(y), 0,
      sin(y), cos(y), 0,
      0, 0, 1;
  Eigen::Matrix<Scalar_t, 3, 3> Ry;
  Ry << cos(p), 0., sin(p),
      0., 1., 0.,
      -sin(p), 0., cos(p);
  Eigen::Matrix<Scalar_t, 3, 3> Rx;
  Rx << 1., 0., 0.,
      0., cos(r), -sin(r),
      0., sin(r), cos(r);

  return Rz * Ry * Rx;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 1> R2ypr(const Eigen::MatrixBase<Derived> &R) {
  typedef typename Derived::Scalar Scalar_t;
  Eigen::Matrix<typename Derived::Scalar, 3, 1> n = R.col(0);
  Eigen::Matrix<typename Derived::Scalar, 3, 1> o = R.col(1);
  Eigen::Matrix<typename Derived::Scalar, 3, 1> a = R.col(2);

  Eigen::Matrix<typename Derived::Scalar, 3, 1> ypr(3);
  Scalar_t y = atan2(n(1), n(0));
  Scalar_t p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
  Scalar_t r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
  ypr(0) = y;
  ypr(1) = p;
  ypr(2) = r;

  return ypr;
}


template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> g2R(const Eigen::MatrixBase<Derived> &g) {
  typedef typename Derived::Scalar Scalar_t;
  Eigen::Matrix<typename Derived::Scalar, 3, 3> R0;
  Eigen::Matrix<typename Derived::Scalar, 3, 1> ng1 = g.normalized();
  Eigen::Matrix<typename Derived::Scalar, 3, 1> ng2{0, 0, 1.0};
  R0 = Eigen::Quaternion<Scalar_t>::FromTwoVectors(ng1, ng2).toRotationMatrix();
  Scalar_t yaw = R2ypr(R0).x();
  R0 = ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
  return R0;
}

template <typename Derived>
static typename Derived::Scalar graffMetric(const Eigen::MatrixBase<Derived> &Ya, const Eigen::MatrixBase<Derived> &Yb) {
  typename Derived::Scalar res;
  
  return res;
}

// void SolveICP(const std::vector<Eigen::Vector3d>& pts1, const std::vector<Eigen::Vector3d>& pts2, Eigen::Matrix3d& R12, Eigen::Vector3d& t12) {
//   assert(pts1.size() == pts2.size());
//   const int n = pts1.size();
//   // step 1. compute center points
//   Eigen::Vector3d pm1(0.0, 0.0, 0.0);
//   Eigen::Vector3d pm2(0.0, 0.0, 0.0);
  
//   for(int i = 0; i < n; ++i) {
//     pm1 += pts1.at(i);
//     pm2 += pts2.at(i);
//   }

//   pm1/= (double)n;
//   pm2/= (double)n;

//   Eigen::Matrix3d W;
//   W.setZero();
//   for(int i = 0; i < n; ++i) {
//     W += (pts1.at(i) - pm1) * (pts2.at(i) - pm2).transpose();
//   }

//   Eigen::JacobiSVD<Eigen::Matrix3d> svd (W, Eigen::ComputeFullU | Eigen::ComputeFullV);
//   Eigen::Matrix3d U = svd.matrixU();
//   Eigen::Matrix3d V = svd.matrixV();

//   R12 = U * V.transpose();

//   if(R12.determinant() < 0) {
//     R12.block(2, 0, 1, 3) = -R12.block(2, 0, 1, 3);
//   }
//   // step 3. compute t
//   t12 = pm1 - R12 * pm2;
// }

  
}

#endif