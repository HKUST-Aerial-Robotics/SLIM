#ifndef PRIOR_POSE_FACTOR_H
#define PRIOR_POSE_FACTOR_H

#include <ceres/ceres.h>

#include <Eigen/Core>

#include "transform.h"
#include "factor/math_utility.h"
namespace SLIM {
class PriorPoseFactor : public ceres::SizedCostFunction<6, 7> {
 public:
  PriorPoseFactor(const Transform& prior_pose, const Eigen::Matrix<double, 6, 6>& sqrt_info)
  : prior_pose_(prior_pose), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

 private:
  Transform prior_pose_;
  Eigen::Matrix<double, 6, 6> sqrt_info_;
};
}
#endif