#ifndef LASER_SURF_FACTOR_H
#define LASER_SURF_FACTOR_H

#include <ceres/ceres.h>

#include <Eigen/Core>

#include "transform.h"
#include "factor/math_utility.h"
#include "factor/surface_info.h"

namespace SLIM {
class LaserPointToPointFactor : public ceres::SizedCostFunction<3, 7> {
 public:
  LaserPointToPointFactor(const Eigen::Vector3d p, const Eigen::Vector3d& q, const double sqrt_info)
  : p_(p), q_(q)
  {
    sqrt_info_ = Eigen::Matrix3d::Identity() * sqrt_info;
  }

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

 private:
  Eigen::Vector3d p_, q_;
  Eigen::Matrix3d sqrt_info_;
};

class LaserPointToSurfaceFactor : public ceres::SizedCostFunction<1, 7> {
 public:
  LaserPointToSurfaceFactor(const Eigen::Vector3d p, const Eigen::Vector3d& q, const Eigen::Vector3d& n, const double sqrt_info)
  : p_(p), q_(q), n_(n), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

 private:
  Eigen::Vector3d p_, q_, n_;
  double sqrt_info_;
};

// class LaserPointToSurfaceGtsamFactor : public gtsam::NoiseModelFactor1<gtsam::Pose3> {
//  public:
//   LaserPointToSurfaceGtsamFactor(const gtsam::SharedNoiseModel &model, gtsam::Key X, const Eigen::Vector3d p, const Eigen::Vector3d& q, const Eigen::Vector3d& n)
//   : NoiseModelFactor1<gtsam::Pose3>(model, X), p_(p), q_(q), n_(n) {}
  
//   gtsam::Vector evaluateError(const gtsam::Pose3 &X, boost::optional<gtsam::Matrix &> H = boost::none) const;

//   /// @return a deep copy of this factor
//   virtual gtsam::NonlinearFactor::shared_ptr clone() const {
//     return boost::static_pointer_cast<gtsam::NonlinearFactor>(
//             gtsam::NonlinearFactor::shared_ptr(new LaserPointToSurfaceGtsamFactor(*this)));
//   }
  
//  private:
//   Eigen::Vector3d p_, q_, n_;
// };


class LaserSurfPriorFactor : public ceres::SizedCostFunction<4, 3> {
 public:
  LaserSurfPriorFactor(const Eigen::Vector3d& pa, const Eigen::Vector3d& pb,
                       const Eigen::Vector3d& pc, const Eigen::Vector3d& pd, 
                       const double& sqrt_info)
  : pa_(pa), pb_(pb), pc_(pc), pd_(pd), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void CheckResidual(double const *const *parameters, double *residuals) const;

  void CheckJacobian(double const *const *parameters);

 private:
  Eigen::Vector3d pa_, pb_, pc_, pd_;
  double sqrt_info_;
};

class LaserSurfFactor : public ceres::SizedCostFunction<1, 7, 3> {
 public:
  LaserSurfFactor(const Eigen::Vector3d& local_point, const double& sqrt_info)
  : local_point_(local_point), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void CheckResidual(double const *const *parameters, double *residuals) const;

  void CheckJacobian(double const *const *parameters);

 private:
  Eigen::Vector3d local_point_;
  double sqrt_info_;
};

class LaserSurf4PFactor : public ceres::SizedCostFunction<4, 7, 3> {
 public:
  LaserSurf4PFactor(const std::vector<Eigen::Vector3d>& vertices, const double& sqrt_info)
  : vertices_(vertices) {
    sqrt_info_ = Eigen::Matrix4d::Identity() * sqrt_info;
  }

  LaserSurf4PFactor(const std::vector<Eigen::Vector3d>& vertices, const Eigen::Matrix4d& sqrt_info)
  : vertices_(vertices), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void CheckJacobian(double const *const *parameters);

 private:
  std::vector<Eigen::Vector3d> vertices_;
  Eigen::Matrix4d sqrt_info_;
};


class LaserSurf3PFactor : public ceres::SizedCostFunction<3, 7, 3> {
 public:
  LaserSurf3PFactor(const std::vector<Eigen::Vector3d>& vertices, const double& sqrt_info)
  : vertices_(vertices) {
    sqrt_info_ = Eigen::Matrix3d::Identity() * sqrt_info;
  }

  LaserSurf3PFactor(const std::vector<Eigen::Vector3d>& vertices, const Eigen::Matrix3d& sqrt_info)
  : vertices_(vertices), sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

  void CheckJacobian(double const *const *parameters);

 private:
  std::vector<Eigen::Vector3d> vertices_;
  Eigen::Matrix3d sqrt_info_;
};


class LaserSurfOnly3PFactor : public ceres::SizedCostFunction<3, 3> {
 public:
  LaserSurfOnly3PFactor(const std::vector<Eigen::Vector3d>& vertices, const Transform& Twb, const double& sqrt_info)
  : vertices_(vertices), Twb_(Twb) {
    sqrt_info_ = Eigen::Matrix3d::Identity() * sqrt_info;
  }

  LaserSurfOnly3PFactor(const std::vector<Eigen::Vector3d>& vertices, const Transform& Twb, const Eigen::Matrix3d& sqrt_info)
  : vertices_(vertices), Twb_(Twb) , sqrt_info_(sqrt_info) {}

  virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

 private:
  std::vector<Eigen::Vector3d> vertices_;
  Transform Twb_;
  Eigen::Matrix3d sqrt_info_;
};


// using namespace gtsam;
// class LaserSurfGtsamFactor : public NoiseModelFactor2<Pose3, Vector3> {
//  public:
//   LaserSurfGtsamFactor(const SharedNoiseModel& noise_model, Key key1, Key key2, const std::vector<Point3>& vertices)
//   : NoiseModelFactor2<Pose3, Vector3>(noise_model, key1, key2), vertices_(vertices) {

//   }

//   virtual Vector evaluateError(const Pose3& pose, const Vector3& lm, boost::optional<Matrix&> H1 = boost::none, boost::optional<Matrix&> H2 = boost::none) const;

//   virtual gtsam::NonlinearFactor::shared_ptr clone() const {
//     return boost::static_pointer_cast<gtsam::NonlinearFactor>(
//             gtsam::NonlinearFactor::shared_ptr(new LaserSurfGtsamFactor(*this)));
//   }

//  private:
//   std::vector<Point3> vertices_;
// };

}

#endif