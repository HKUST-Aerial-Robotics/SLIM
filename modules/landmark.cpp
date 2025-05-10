#include "landmark.h"
#include <ceres/ceres.h>
#include "factor/laser_edge_factor.h"
#include "factor/laser_surf_factor.h"
#include "factor/local_param.h"

namespace SLIM {

bool LineLM::associate(LineOB::Ptr const& ob, Transform const& Twb) {
  Eigen::Vector3d const fn = Twb.dcm() * ob->normal();
  double theta = std::acos(fn.dot(normal_));
  if(theta > M_PI/36.0 && theta < M_PI - M_PI/36.0) {
    return false;
  }

  Eigen::Matrix3d const nmat = Eigen::Matrix3d::Identity() - normal_ * normal_.transpose();
  Eigen::Vector3d const dist_a = nmat * (Twb * ob->point_a() - centroid_);
  Eigen::Vector3d const dist_b = nmat * (Twb * ob->point_b() - centroid_);
  if(dist_a.norm() > 0.2 || dist_b.norm() > 0.2) {
    return false;
  }
  return true;
}

void LineLM::solveByObs() {
  if(!updated_)
    return;
  
  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d squared_sum{Eigen::Matrix3d::Zero()};
  int size = 0;
  for(auto &ob_iter: obvs_) {
    Frame::Ptr const f = ob_iter.first;
    LineOB::Ptr const ob = ob_iter.second;
    Transform T = f->constTwb();
    auto gpa = T * ob->point_a(), gpb = T * ob->point_b();
    
    sum += gpa * ob->point_num_ * 0.5;
    sum += gpb * ob->point_num_ * 0.5;
    squared_sum += gpa * gpa.transpose() * ob->point_num_ * 0.5;
    squared_sum += gpb * gpb.transpose() * ob->point_num_ * 0.5;
    size += ob->point_num_;
  }
  
  sum /= size;
  squared_sum.noalias() = squared_sum / size - sum * sum.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(squared_sum);
  Eigen::Vector3d lambda = saes.eigenvalues();
  normal_ = saes.eigenvectors().col(2);
  centroid_ = sum;

  if(centroid_.array().isNaN().any() || centroid_.array().isInf().any()) {
    printf("Error NAN & INF Values\n");
    std::cout << "sum: " << sum.transpose() << std::endl;
    std::cout << "squared_sum: " << std::endl << squared_sum << std::endl;
    std::cout << "point_num: " << size << std::endl;
    for(int i = 0; i < obvs_.size(); ++i) {
      std::cout << "ob: " << i << " point_num: " << obvs_[i].second->point_num_ << std::endl;
    }
  }
  vector2double();
  pa_ = centroid_ + normal_ * std::sqrt(lambda(2) * 2);
  pb_ = centroid_ - normal_ * std::sqrt(lambda(2) * 2);
  updated_ = false;
}

void LineLM::double2vector() {

  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d squared_sum{Eigen::Matrix3d::Zero()};
  int size = 0;
  for(auto &ob_iter: obvs_) {
    Frame::Ptr const f = ob_iter.first;
    LineOB::Ptr const ob = ob_iter.second;
    Transform T = f->constTwb();
    auto gpa = T * ob->point_a(), gpb = T * ob->point_b();
    
    sum += gpa * ob->point_num_ * 0.5;
    sum += gpb * ob->point_num_ * 0.5;
    squared_sum += gpa * gpa.transpose() * ob->point_num_ * 0.5;
    squared_sum += gpb * gpb.transpose() * ob->point_num_ * 0.5;
    size += ob->point_num_;
  }
  
  sum /= size;
  squared_sum.noalias() = squared_sum / size - sum * sum.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(squared_sum);
  Eigen::Vector3d lambda = saes.eigenvalues();
  normal_ = line_.get_normal();
  centroid_ = line_.pedal(sum);
  auto r = std::sqrt(lambda(2) * 2);
  pa_ = centroid_ + normal_ * r;
  pb_ = centroid_ - normal_ * r;
}


bool SurfaceLM::associate(SurfaceOB::Ptr const& ob, Transform const& Twb) {
  Eigen::Vector3d const fn = Twb.dcm() * ob->normal();
  Eigen::Vector3d const fc = Twb * ob->centroid();
  if((fc - centroid_).norm() > 2.0 * radius_) {
    return false;
  }

  double theta = std::acos(fn.dot(normal_));
  if(theta > M_PI/36.0 && theta < M_PI - M_PI/36.0) {
    return false;
  }

  std::vector<Eigen::Vector3d> const vertices = ob->vertices();
  double dist_a = normal_.dot(Twb * vertices[0] - centroid_);
  double dist_b = normal_.dot(Twb * vertices[1] - centroid_);
  double dist_c = normal_.dot(Twb * vertices[2] - centroid_);
  if(std::abs(dist_a) > 0.2 || std::abs(dist_b) > 0.2 || std::abs(dist_c) > 0.2) {
    return false;
  }

  return true;
}

void SurfaceLM::solveByObs() {

  if(!updated_)
    return;

  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d squared_sum{Eigen::Matrix3d::Zero()};
  int size = 0;
  for(auto &ob_iter: obvs_) {
    Frame::Ptr const f = ob_iter.first;
    SurfaceOB::Ptr const ob = ob_iter.second;
    Transform T = f->constTwb();

    auto gpa = T * ob->vertices()[0], gpb = T * ob->vertices()[1], gpc = T * ob->vertices()[2];
    sum += gpa * ob->point_num_ / 3.0;
    sum += gpb * ob->point_num_ / 3.0;
    sum += gpc * ob->point_num_ / 3.0;
    squared_sum += gpa * gpa.transpose() * ob->point_num_ / 3.0;
    squared_sum += gpb * gpb.transpose() * ob->point_num_ / 3.0;
    squared_sum += gpc * gpc.transpose() * ob->point_num_ / 3.0;
    size += ob->point_num_;
  }
  
  sum /= size;
  squared_sum.noalias() = squared_sum / size - sum * sum.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(squared_sum);
  Eigen::Vector3d lambda = saes.eigenvalues();
  Eigen::Matrix3d umat = saes.eigenvectors();
  normal_ = saes.eigenvectors().col(0);
  radius_ = std::sqrt(lambda(1) * 2);
  centroid_ = sum;
  if(centroid_.array().isNaN().any() || centroid_.array().isInf().any()) {
    printf("Error NAN & INF Values\n");
    std::cout << "sum: " << sum.transpose() << std::endl;
    std::cout << "squared_sum: " << std::endl << squared_sum << std::endl;
    std::cout << "point_num: " << size << std::endl;
    for(int i = 0; i < obvs_.size(); ++i) {
      std::cout << "ob: " << i << " point_num: " << obvs_[i].second->point_num_ << std::endl;
    }
  }
  vector2double();

  double ra = std::sqrt(lambda(2) * 2);
  double rb = std::sqrt(lambda(1) * 2);

  vertices_.resize(4);
  vertices_[0] = centroid_ + umat.col(2) * ra;
  vertices_[1] = centroid_ - umat.col(2) * ra;
  vertices_[2] = centroid_ + umat.col(1) * rb;
  vertices_[3] = centroid_ - umat.col(1) * rb;
  updated_ = false;
}
  
void SurfaceLM::double2vector() {

  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d squared_sum{Eigen::Matrix3d::Zero()};
  int size = 0;
  for(auto &ob_iter: obvs_) {
    Frame::Ptr const f = ob_iter.first;
    SurfaceOB::Ptr const ob = ob_iter.second;
    Transform T = f->constTwb();
    
    auto gpa = T * ob->vertices()[0], gpb = T * ob->vertices()[1], gpc = T * ob->vertices()[2];
    sum += gpa * ob->point_num_ / 3.0;
    sum += gpb * ob->point_num_ / 3.0;
    sum += gpc * ob->point_num_ / 3.0;
    squared_sum += gpa * gpa.transpose() * ob->point_num_ / 3.0;
    squared_sum += gpb * gpb.transpose() * ob->point_num_ / 3.0;
    squared_sum += gpc * gpc.transpose() * ob->point_num_ / 3.0;
    size += ob->point_num_;
  }
  
  sum /= size;
  squared_sum.noalias() = squared_sum / size - sum * sum.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(squared_sum);
  Eigen::Vector3d lambda = saes.eigenvalues();
  Eigen::Matrix3d umat = saes.eigenvectors();
  double ra = std::sqrt(lambda(2) * 2);
  double rb = std::sqrt(lambda(1) * 2);

  normal_ = surface_.get_normal();
  centroid_ = surface_.pedal(sum);
  radius_ = std::sqrt(lambda(1) * 2);

  Eigen::Matrix3d R = Eigen::Quaterniond::FromTwoVectors(umat.col(0), normal_).toRotationMatrix();
  Eigen::Matrix3d dir = R * umat;

  vertices_.resize(4);
  vertices_[0] = centroid_ + dir.col(2) * ra;
  vertices_[1] = centroid_ - dir.col(2) * ra;
  vertices_[2] = centroid_ + dir.col(1) * rb;
  vertices_[3] = centroid_ - dir.col(1) * rb;
}


} // namespace SLIM
