#include "factor/local_param.h"

namespace SLIM {
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const {
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);
  Eigen::Map<const Eigen::Vector3d> dp(delta);
  Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));
  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);
  p = _p + dp;
  q = (_q * dq).normalized();
  return true;
}

bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
  J.topRows<6>().setIdentity();
  J.bottomRows<1>().setZero();
  return true;
}


bool LineLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const {
  Eigen::Map<const Eigen::Vector4d> psi(x);
  Eigen::Map<const Eigen::Vector4d> dpsi(delta);
  Eigen::Map<Eigen::Vector4d> psi_plus_dpsi(x_plus_delta);
  psi_plus_dpsi = psi + dpsi;

  if(psi_plus_dpsi(0) >= M_PI) {
    psi_plus_dpsi(0) = psi_plus_dpsi(0) - 2 * M_PI;
  }
  else if(psi_plus_dpsi(0) < -M_PI) {
    psi_plus_dpsi(0) = psi_plus_dpsi(0) + 2 * M_PI;
  }

  if(psi_plus_dpsi(1) >= M_PI) {
    psi_plus_dpsi(1) = psi_plus_dpsi(1) - 2 * M_PI;
  }
  else if(psi_plus_dpsi(1) < -M_PI) {
    psi_plus_dpsi(1) = psi_plus_dpsi(1) + 2 * M_PI;
  }
  return true;
}

bool LineLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> J(jacobian);
  J.setIdentity();
  return true;
}

bool SurfaceLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const {
  Eigen::Map<const Eigen::Vector3d> psi(x);
  Eigen::Map<const Eigen::Vector3d> dpsi(delta);
  Eigen::Map<Eigen::Vector3d> psi_plus_dpsi(x_plus_delta);
  psi_plus_dpsi = psi + dpsi;

  if(psi_plus_dpsi(0) >= M_PI) {
    psi_plus_dpsi(0) = psi_plus_dpsi(0) - 2 * M_PI;
  }
  else if(psi_plus_dpsi(0) < -M_PI) {
    psi_plus_dpsi(0) = psi_plus_dpsi(0) + 2 * M_PI;
  }

  if(psi_plus_dpsi(1) >= M_PI) {
    psi_plus_dpsi(1) = psi_plus_dpsi(1) - 2 * M_PI;
  }
  else if(psi_plus_dpsi(1) < -M_PI) {
    psi_plus_dpsi(1) = psi_plus_dpsi(1) + 2 * M_PI;
  }
  return true;
}

bool SurfaceLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const {
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobian);
  J.setIdentity();
  return true;
}
}

