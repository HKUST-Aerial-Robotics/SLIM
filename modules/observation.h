#ifndef OBSERVATION_H
#define OBSERVATION_H

#include <mutex>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <opencv2/opencv.hpp>

#include "transform.h"
#include "semantic_definition.h"

namespace SLIM {

// pole like feature
class LineOB {
 public:
  typedef std::shared_ptr<LineOB> Ptr;
  LineOB() = default;

  LineOB(const uint16_t semantic_type)
  : semantic_type_(semantic_type) {}

  LineOB(const uint16_t semantic_type, const Eigen::Vector3d& pa, const Eigen::Vector3d& pb)
  : semantic_type_(semantic_type), point_a_(pa), point_b_(pb), is_active_(true) {

  }

  LineOB(const uint16_t semantic_type, const Eigen::Vector3d& sum, const Eigen::Matrix3d& squared_sum, const uint32_t point_num);

  LineOB(const uint16_t semantic_type, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

  LineOB(const uint16_t semantic_type, const Eigen::Vector3d& point_a, const Eigen::Vector3d& point_b, const uint32_t point_num);

  bool Init(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

  void setSqrtInfo(const Eigen::Matrix4d& sqrt_info) {
    sqrt_info_ = sqrt_info;
  }

  void SetActive() {
    is_active_ = true;
  }

  void SetInactive() {
    is_active_ = false;
  }

  // double lambda() const {
  //   return lambda_(2);
  // }

  Eigen::Vector3d centroid() const {
    return 0.5 * (point_a_ + point_b_);
  }

  Eigen::Vector3d normal() const {
    return (point_a_ - point_b_).normalized();
  }

  Eigen::Vector3d point_a() const {
    return point_a_;
  }

  Eigen::Vector3d point_b() const {
    return point_b_;
  }

  // Eigen::Matrix3d sigma() const {
  //   return sigma_;
  // }

  uint16_t semantic_type() const {
    return semantic_type_;
  }

  const Eigen::Matrix4d sqrt_info() const {
    return sqrt_info_;
  }

 public:
  Eigen::Vector3d point_a_, point_b_;
  Eigen::Vector3d sum_;
  Eigen::Matrix3d squared_sum_;
  Eigen::Matrix3d sigma_;
  Eigen::Vector3d lambda_;
  Eigen::Matrix4d sqrt_info_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
  uint32_t point_num_;
  uint16_t semantic_type_;
  bool is_active_ = false;
  std::mutex data_mutex_;
};

// road, sidewalk, wall, etc.
class SurfaceOB {
 public:
  typedef std::shared_ptr<SurfaceOB> Ptr;
  SurfaceOB() = default;

  SurfaceOB(const uint16_t semantic_type)
  : semantic_type_(semantic_type) {}

  SurfaceOB(const uint16_t semantic_type, const Eigen::Vector3d& center, 
                    const Eigen::Vector3d& normal, const std::vector<Eigen::Vector3d>& vertices,
                    const float ra, const float rb);

  SurfaceOB(const uint16_t semantic_type, const Eigen::Vector3d& sum, const Eigen::Matrix3d& squared_sum, const uint32_t point_num);

  SurfaceOB(const uint16_t semantic_type, const std::vector<Eigen::Vector3d>& vertices, const uint32_t point_num);

  SurfaceOB(const uint16_t semantic_type, const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud);

  // SurfaceOB(const uint16_t semantic_type, const Eigen::Vector3d& center, const Eigen::Vector3d& normal, const float ra, const float rb);
  // bool Init(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  //   int N = cloud->size();
  //   sigma_.setZero();
  //   center_.setZero();
  //   for(int i = 0; i < N; ++i) {
  //     Eigen::Vector3d point = cloud->points[i].getVector3fMap().cast<double>();
  //     center_ += point;
  //     sigma_ += point * point.transpose();
  //   }
  //   center_ /= N;
  //   sigma_.noalias() = sigma_ / N - center_ * center_.transpose();

  //   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma_);
  //   lambda_ = saes.eigenvalues();

  //   if(lambda_(0) > 0.04 * lambda_(1) || lambda_(1) < 0.2 * lambda_(2)) {
  //     return false;
  //   }

  //   umat_ = saes.eigenvectors();
  //   normal_ = umat_.col(0);
  //   return true;
  // }

  void insertParticle(const Eigen::Vector3d& particle) {
    particles.push_back(particle);
  }

  void setSqrtInfo(const Eigen::Matrix3d& sqrt_info) {
    sqrt_info_ = sqrt_info;
  }

  // std::vector<Eigen::Vector3d> vertices() const {
  //   // std::unique_lock<std::mutex> lock(data_mutex_);
  //   std::vector<Eigen::Vector3d> vertices(4);
  //   vertices[0] = center_ + umat_.col(1) * std::sqrt(lambda_(1) * 2);
  //   vertices[1] = center_ + umat_.col(2) * std::sqrt(lambda_(2) * 2);
  //   vertices[2] = center_ - umat_.col(1) * std::sqrt(lambda_(1) * 2);
  //   vertices[3] = center_ - umat_.col(2) * std::sqrt(lambda_(2) * 2);
  //   // vertices[0] = center_ + umat_.col(1) * (lambda_(1));
  //   // vertices[1] = center_ + umat_.col(2) * (lambda_(2));
  //   // vertices[2] = center_ - umat_.col(1) * (lambda_(1));
  //   // vertices[3] = center_ - umat_.col(2) * (lambda_(2));
  //   return vertices;
  // }

  std::vector<Eigen::Vector3d> vertices() const {
    // std::unique_lock<std::mutex> lock(data_mutex_);
    return vertices_;
  }

  uint16_t semantic_type() const {
    // std::unique_lock<std::mutex> lock(data_mutex_);
    return semantic_type_;
  }

  void SetActive() {
    is_active_ = true;
  }

  void SetInactive() {
    is_active_ = false;
  }

  // double lambda() const {
  //   return lambda_(0);
  // }

  float ra() const {
    return ra_;
  }

  float rb() const {
    return rb_;
  }

  Eigen::Vector3d centroid() const {
    return center_;
  }

  Eigen::Vector3d normal() const {
    return normal_;
  }

  // Eigen::Matrix3d sigma() const {
  //   return sigma_;
  // }

  const Eigen::Matrix3d sqrt_info() const {
    return sqrt_info_;
  }

 public:
  Eigen::Vector3d center_, normal_;
  Eigen::Vector3d sum_;
  Eigen::Matrix3d squared_sum_;
  // Eigen::Matrix3d sigma_;
  // Eigen::Vector3d lambda_;
  // Eigen::Matrix3d umat_;
  Eigen::Matrix3d sqrt_info_;
  std::vector<Eigen::Vector3d> particles;
  std::vector<Eigen::Vector3d> vertices_;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
  uint32_t point_num_;
  uint16_t semantic_type_;
  float ra_, rb_;
  bool is_active_ = false;
  std::mutex data_mutex_;
};

} // namespace SLIM


#endif