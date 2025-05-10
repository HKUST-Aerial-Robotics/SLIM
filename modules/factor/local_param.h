#ifndef LOCAL_PARAM_H
#define LOCAL_PARAM_H

#include <ceres/ceres.h>

#include <Eigen/Core>

#include "transform.h"
#include "factor/math_utility.h"

namespace SLIM {
class PoseLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 7; };
  virtual int LocalSize() const { return 6; };
};

class LineLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 4; };
  virtual int LocalSize() const { return 4; };
};

class SurfaceLocalParameterization : public ceres::LocalParameterization {
  virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
  virtual bool ComputeJacobian(const double *x, double *jacobian) const;
  virtual int GlobalSize() const { return 3; };
  virtual int LocalSize() const { return 3; };
};
}

#endif

