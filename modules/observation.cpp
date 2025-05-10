#include "observation.h"

namespace SLIM {

bool LineOB::Init(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  int N = cloud->size();
  sigma_.setZero();
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  for(int i = 0; i < N; ++i) {
    Eigen::Vector3d point = cloud->points[i].getVector3fMap().cast<double>();
    center += point;
    sigma_ += point * point.transpose();
  }
  center /= N;
  sigma_.noalias() = sigma_ / N - center * center.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma_);
  lambda_ = saes.eigenvalues();
  if(lambda_(1) > 0.05 * lambda_(2)) {
    return false;
  }

  Eigen::Matrix3d umat = saes.eigenvectors();
  Eigen::Vector3d normal = umat.col(2);
  point_a_ = center + normal * std::sqrt(lambda_(2) * 2);
  point_b_ = center - normal * std::sqrt(lambda_(2) * 2);
  sqrt_info_ = Eigen::Matrix4d::Identity() * std::sqrt(lambda_(1) * 2);
  return true;
} 

LineOB::LineOB(const uint16_t semantic_type, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  point_num_ = cloud->size();
  semantic_type_ = semantic_type;
  cloud_ = cloud;
  sum_.setZero();
  squared_sum_.setZero();
  for(int i = 0; i < point_num_; ++i) {
    Eigen::Vector3d point = cloud->points[i].getVector3fMap().cast<double>();
    sum_ += point;
    squared_sum_ += point * point.transpose();
  }
  Eigen::Vector3d center = sum_ / point_num_;
  sigma_ = squared_sum_ / point_num_ - center * center.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma_);
  lambda_ = saes.eigenvalues();

  Eigen::Matrix3d umat = saes.eigenvectors();
  Eigen::Vector3d normal = umat.col(2);
  point_a_ = center + normal * std::sqrt(lambda_(2) * 2);
  point_b_ = center - normal * std::sqrt(lambda_(2) * 2);
  sqrt_info_ = Eigen::Matrix4d::Identity() * std::sqrt(point_num_ / 2) / (0.2);
}

// LineOB::LineOB(const uint16_t semantic_type, const Eigen::Vector3d& sum, const Eigen::Matrix3d& squared_sum, const uint32_t point_num) {
//   semantic_type_ = semantic_type;
//   point_num_ = point_num;
//   sum_ = sum;
//   squared_sum_ = squared_sum;

//   Eigen::Vector3d center = sum / point_num_;
//   sigma_ = squared_sum_ / point_num_ - center * center.transpose();
//   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma_);
//   lambda_ = saes.eigenvalues();
//   Eigen::Vector3d normal = saes.eigenvectors().col(2);
//   point_a_ = center + normal * std::sqrt(lambda_(2) * 2);
//   point_b_ = center - normal * std::sqrt(lambda_(2) * 2);
// }

LineOB::LineOB(const uint16_t semantic_type, const Eigen::Vector3d& point_a, const Eigen::Vector3d& point_b, const uint32_t point_num)
: semantic_type_(semantic_type), point_a_(point_a), point_b_(point_b), point_num_(point_num) {}

// SurfaceOB::SurfaceOB(const uint16_t semantic_type, const Eigen::Vector3d& center, 
//                  const Eigen::Vector3d& normal, const float ra, const float rb)
// : semantic_type_(semantic_type), center_(center), normal_(normal), sigma_(sigma), ra_(ra), rb_(rb) {
//   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma_);
//   lambda_ = saes.eigenvalues();
//   umat_ = saes.eigenvectors();
//   normal_ = umat_.col(0);
// }

SurfaceOB::SurfaceOB(const uint16_t semantic_type, const Eigen::Vector3d& center, 
                    const Eigen::Vector3d& normal, const std::vector<Eigen::Vector3d>& vertices,
                    const float ra, const float rb)
: semantic_type_(semantic_type), center_(center), normal_(normal), vertices_(vertices), ra_(ra), rb_(rb) {
  
}

SurfaceOB::SurfaceOB(const uint16_t semantic_type, const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
  point_num_ = cloud->size();
  semantic_type_ = semantic_type;
  cloud_ = cloud;
  sum_.setZero();
  squared_sum_.setZero();
  for(int i = 0; i < point_num_; ++i) {
    Eigen::Vector3d point = cloud->points[i].getVector3fMap().cast<double>();
    sum_ += point;
    squared_sum_ += point * point.transpose();
  }
  center_ = sum_ / point_num_;
  Eigen::Matrix3d sigma = squared_sum_ / point_num_ - center_ * center_.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma);
  Eigen::Vector3d lambda = saes.eigenvalues();
  Eigen::Matrix3d umat = saes.eigenvectors();
  normal_ = umat.col(0);

  ra_ = std::sqrt(lambda(2) * 2);
  rb_ = std::sqrt(lambda(1) * 2);

  vertices_.resize(3);
  vertices_[0] = center_ + umat.col(2) * ra_;
  vertices_[1] = center_ - umat.col(2) * ra_ * 0.5 + umat.col(1) * rb_ * 0.5;
  vertices_[2] = center_ - umat.col(2) * ra_ * 0.5 - umat.col(1) * rb_ * 0.5;
  if(semantic_type_ == ROAD_ID) {
    sqrt_info_ = Eigen::Matrix3d::Identity() * std::sqrt(point_num_ / 3) / (0.1);
  }
  else {
    sqrt_info_ = Eigen::Matrix3d::Identity() * std::sqrt(point_num_ / 3) / (0.2);
  }

  if(vertices_[0].array().isNaN().any() || vertices_[0].array().isInf().any()) {
    printf("Error NAN & INF Values, v0 \n");
    std::cout << "sigma: " << std::endl << sigma << std::endl;
    std::cout << "lambda: " << std::endl << lambda << std::endl;
    std::cout << "num: " << std::endl << point_num_ << std::endl;
    std::cout << "num: " << std::endl << point_num_ << std::endl;
  } 
  if(vertices_[1].array().isNaN().any() || vertices_[1].array().isInf().any()) {
    printf("Error NAN & INF Values, v1 \n");
    std::cout << "sigma: " << std::endl << sigma << std::endl;
    std::cout << "lambda: " << std::endl << lambda << std::endl;
    std::cout << "num: " << std::endl << point_num_ << std::endl;
    std::cout << "num: " << std::endl << point_num_ << std::endl;
  } 
  if(vertices_[2].array().isNaN().any() || vertices_[2].array().isInf().any()) {
    printf("Error NAN & INF Values, v2 \n");
    std::cout << "sigma: " << std::endl << sigma << std::endl;
    std::cout << "lambda: " << std::endl << lambda << std::endl;
    std::cout << "num: " << std::endl << point_num_ << std::endl;
    std::cout << "num: " << std::endl << point_num_ << std::endl;
  } 
  
}

SurfaceOB::SurfaceOB(const uint16_t semantic_type, const std::vector<Eigen::Vector3d>& vertices, const uint32_t point_num)
: semantic_type_(semantic_type), vertices_(vertices), point_num_(point_num) {

}

// SurfaceOB::SurfaceOB(const uint16_t semantic_type, const Eigen::Vector3d& sum, const Eigen::Matrix3d& squared_sum, const uint32_t point_num) {
//   semantic_type_ = semantic_type;
//   point_num_ = point_num;
//   sum_ = sum;
//   squared_sum_ = squared_sum;

//   center_ = sum_ / point_num_;
//   Eigen::Matrix3d sigma = squared_sum_ / point_num_ - center_ * center_.transpose();

//   Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma);
//   Eigen::Vector3d lambda = saes.eigenvalues();
//   Eigen::Matrix3d umat = saes.eigenvectors();
//   normal_ = umat.col(0);

//   ra_ = std::sqrt(lambda(2) * 2);
//   rb_ = std::sqrt(lambda(1) * 2);


//   if(lambda.array().isNaN().any() || lambda.array().isInf().any()) {
//     printf("Error NAN & INF Values, lambda \n");
//     std::cout << "sigma: " << std::endl << sigma << std::endl;
//     std::cout << "lambda: " << std::endl << lambda << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//   }          
//   if(center_.array().isNaN().any() || center_.array().isInf().any()) {
//     printf("Error NAN & INF Values, center \n");
//   }      
//   if(umat.array().isNaN().any() || umat.array().isInf().any()) {
//     printf("Error NAN & INF Values, center \n");
//   }     

//   vertices_.resize(3);
//   vertices_[0] = center_ + umat.col(2) * ra_;
//   vertices_[1] = center_ - umat.col(2) * ra_ * 0.5 + umat.col(1) * rb_ * 0.5;
//   vertices_[2] = center_ - umat.col(2) * ra_ * 0.5 - umat.col(1) * rb_ * 0.5;
//   if(vertices_[0].array().isNaN().any() || vertices_[0].array().isInf().any()) {
//     printf("Error NAN & INF Values, v0 \n");
//     std::cout << "sigma: " << std::endl << sigma << std::endl;
//     std::cout << "lambda: " << std::endl << lambda << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//   } 
//   if(vertices_[1].array().isNaN().any() || vertices_[1].array().isInf().any()) {
//     printf("Error NAN & INF Values, v1 \n");
//     std::cout << "sigma: " << std::endl << sigma << std::endl;
//     std::cout << "lambda: " << std::endl << lambda << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//   } 
//   if(vertices_[2].array().isNaN().any() || vertices_[2].array().isInf().any()) {
//     printf("Error NAN & INF Values, v2 \n");
//     std::cout << "sigma: " << std::endl << sigma << std::endl;
//     std::cout << "lambda: " << std::endl << lambda << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//     std::cout << "num: " << std::endl << point_num_ << std::endl;
//   } 
// }


} // namespace SLIM
