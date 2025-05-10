#ifndef LASER_EDGE_FACTOR_H
#define LASER_EDGE_FACTOR_H

#include <ceres/ceres.h>
#include <Eigen/Core>
#include "transform.h"
#include "factor/math_utility.h"
#include "factor/line_info.h"

namespace SLIM {
class LaserPointToLineFactor : public ceres::SizedCostFunction<3, 7> {
 public:
  LaserPointToLineFactor(const Eigen::Vector3d p, const Eigen::Vector3d& q, const Eigen::Vector3d& n, const double sqrt_info)
  : p_(p), q_(q), n_(n), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

 private:
  Eigen::Vector3d p_, q_, n_;
  double sqrt_info_;
};

class LaserEdgePriorFactor : public ceres::SizedCostFunction<6, 4> {
 public:
  LaserEdgePriorFactor(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb, const double sqrt_info)
  : pa_(pa), pb_(pb), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void CheckJacobian(double const *const *parameters);

 private:
  Eigen::Vector3d pa_, pb_;
  double sqrt_info_;
  
};

class LaserEdgeFactor : public ceres::SizedCostFunction<2, 7, 4> {
 public:
  LaserEdgeFactor(const Eigen::Vector3d& local_point, const double& sqrt_info)
  : local_point_(local_point), sqrt_info_(Eigen::Matrix2d::Identity() * sqrt_info) {}

  LaserEdgeFactor(const Eigen::Vector3d& local_point, const Eigen::Matrix2d& sqrt_info)
  : local_point_(local_point), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void CheckJacobian(double const *const *parameters);

 private:
  Eigen::Vector3d local_point_;
  Eigen::Matrix2d sqrt_info_;
};

class LaserEdge2PFactor : public ceres::SizedCostFunction<4, 7, 4> {
 public:
  LaserEdge2PFactor(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb, const double& sqrt_info)
  : pa_(pa), pb_(pb), sqrt_info_(Eigen::Matrix4d::Identity() * sqrt_info) {}

  LaserEdge2PFactor(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb, const Eigen::Matrix4d& sqrt_info)
  : pa_(pa), pb_(pb), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void CheckJacobian(double const *const *parameters);

 private:
  Eigen::Vector3d pa_, pb_;
  Eigen::Matrix4d sqrt_info_;
};


class LaserEdgeOnly2PFactor : public ceres::SizedCostFunction<4, 4> {
 public:
  LaserEdgeOnly2PFactor(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb, const Transform& Twb, const double& sqrt_info)
  : pa_(pa), pb_(pb), Twb_(Twb), sqrt_info_(Eigen::Matrix4d::Identity() * sqrt_info) {}

  LaserEdgeOnly2PFactor(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb, const Transform& Twb, const Eigen::Matrix4d& sqrt_info)
  : pa_(pa), pb_(pb), Twb_(Twb), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

 private:
  Eigen::Vector3d pa_, pb_;
  Transform Twb_;
  Eigen::Matrix4d sqrt_info_;
};

}

#endif