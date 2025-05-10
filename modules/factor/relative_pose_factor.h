#ifndef RELATIVE_POSE_FACTOR_H
#define RELATIVE_POSE_FACTOR_H

#include <ceres/ceres.h>

#include <Eigen/Core>

#include "transform.h"
#include "factor/math_utility.h"
namespace SLIM {
class RelativePoseFactor : public ceres::SizedCostFunction<6, 7, 7> {
 public:
  RelativePoseFactor(const Transform& rel_pose, const Eigen::Matrix<double, 6, 6>& sqrt_info)
  : rel_pose_(rel_pose), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

 private:
  Transform rel_pose_;
  Eigen::Matrix<double, 6, 6> sqrt_info_;
};
}

#endif