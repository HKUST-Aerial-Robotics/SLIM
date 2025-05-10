#include "voxel.h"
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/transforms.h>

namespace SLIM {

VoxelLoc getVoxelLoc(const pcl::PointXYZI& point, const Eigen::Vector3f& voxel_size) {
  Eigen::Vector3f voxel_loc;
  for (int j = 0; j < 3; j++) {
    voxel_loc(j) = point.data[j] / voxel_size(j);
    if (voxel_loc(j) < 0) {
      voxel_loc(j) -= 1.0;
    }
  }
  return VoxelLoc((int64_t)voxel_loc.x(), (int64_t)voxel_loc.y(), (int64_t)voxel_loc.z());
}

void cutCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud, const Transform& Twl, const uint16_t semantic_type,
              const Eigen::Vector3f& voxel_size, std::unordered_map<VoxelLoc, Voxel::Ptr>& voxel_map) {
  Eigen::Matrix4f Twl_mat = Twl.matrix().cast<float>();
  for (size_t i = 0; i < cloud.size(); i++) {
    const pcl::PointXYZI &point = cloud.points[i];
    pcl::PointXYZI global_point = point;
    global_point.getVector4fMap() = Twl_mat * point.getVector4fMap();
    VoxelLoc position = getVoxelLoc(global_point, voxel_size);
    std::unordered_map<VoxelLoc, Voxel::Ptr>::iterator voxel_iter = voxel_map.find(position);
    if(voxel_iter != voxel_map.end()) {
      voxel_iter->second->insertPoint(global_point);
    }
    else {
      Eigen::Vector3d origin(position.x * voxel_size.x(), position.y * voxel_size.y(), position.z * voxel_size.z());
      Voxel::Ptr voxel = Voxel::Ptr(new Voxel(voxel_size, origin, semantic_type));
      voxel->insertPoint(global_point);
      voxel_map.insert(std::make_pair(position, voxel));
    }
  }
}

bool Voxel::parse(const int mode) {

  if(semantic_type_ == ROAD_ID || semantic_type_ == SIDEWALK_ID) {
    pcl::VoxelGrid<pcl::PointXYZI> vf;
    vf.setLeafSize(0.4, 0.4, 0.4);
    vf.setInputCloud(cloud_.makeShared());
    vf.filter(cloud_);
  }

  if(cloud_.size() < 5) {
    return false;
  }
  int N = cloud_.size();

  sigma_.setZero();
  for(int i = 0; i < N; ++i) {
    Eigen::Vector3d pt = cloud_.points[i].getVector3fMap().cast<double>();
    center_ += pt;
    sigma_ += pt * pt.transpose();
  }
  center_ /= N;
  sigma_.noalias() = sigma_ / N - center_ * center_.transpose();

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma_);
  lambda_ = saes.eigenvalues();

  if(lambda_(0) > 0.2 * lambda_(1) || lambda_(1) < 0.4 * lambda_(2)) {
  // if(lambda_(0) > 0.1 * lambda_(1)) {
    return false;
  }
  normal_ = saes.eigenvectors().col(0);
  return true;
}

int OctoVoxel::max_level_ = 4;

bool OctoVoxel::checkPCA(const double thres) {

  pcl::SACSegmentation<pcl::PointXYZI> seg;
  seg.setOptimizeCoefficients(true);
  seg.setModelType(pcl::SACMODEL_PLANE);
  seg.setMethodType(pcl::SAC_RANSAC);
  seg.setMaxIterations(100);
  seg.setDistanceThreshold(thres);

  pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
  seg.setInputCloud(cloud_.makeShared());
  seg.segment(*inliers, *coefficients);

  if ((double)inliers->indices.size() / cloud_.size() < 0.7 || inliers->indices.size() < 20) {
    center_.setZero();
    normal_.setZero();
    return false;
  }

  pcl::PointCloud<pcl::PointXYZI> inlier_cloud;
  for (size_t i = 0; i < inliers->indices.size(); ++i) {
    pcl::PointXYZI point = cloud_.points[inliers->indices[i]];
    inlier_cloud.points.push_back(point);
  }

  center_.setZero();
  sigma_.setZero();
  int N = inlier_cloud.size();
  for(int i = 0; i < N; ++i) {
    Eigen::Vector3f pt = inlier_cloud.points[i].getVector3fMap();
    center_ += pt;
    sigma_ += pt * pt.transpose();
  }
  center_ /= N;
  sigma_.noalias() = sigma_ / N - center_ * center_.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> saes(sigma_);
  lambda_ = saes.eigenvalues();

  float local_radius = std::sqrt(lambda_(1) * 2);
  if(lambda_(1) > 0.2 * lambda_(2)) {
    normal_ = saes.eigenvectors().col(0);
    cloud_.swap(inlier_cloud);
    return true;
  }
  else {
    center_.setZero();
    normal_.setZero();
    return false;
  }

}

void OctoVoxel::parse(const double thres) {
  
  if(!is_updated_)
    return;

  if(is_leaf_) {
    if(cloud_.size() < 10) {
      is_leaf_ = true;
      is_updated_ = false;
      is_plane_ = false;
      return;
    }

    if(checkPCA(thres)) {
      is_leaf_ = true;
      is_plane_ = true;
      is_updated_ = false;
      return;
    }
    else {
      // if(level_ == 0)
      //   std::cout << "surf check failed!" << std::endl;
      is_updated_ = false;
      is_leaf_ = false;        
      is_plane_ = false;
    }
  }

  int next_level = level_ + 1;
  if(next_level == max_level_) {
    return;
  }
  Eigen::Vector3f next_voxel_size = voxel_size_/2;
  int cloud_size = cloud_.size();
  Eigen::Vector3f voxel_center = origin_ + voxel_size_ * 0.5f; 
  for(int i = 0; i < cloud_size; i++) {
    int xyz[3] = {0, 0, 0};
    auto point = cloud_.points[i].getVector3fMap();

    for(int dim = 0; dim < 3; dim++)
      if(point(dim) > voxel_center(dim))
        xyz[dim] = 1;

    int child_id = 4*xyz[0] + 2*xyz[1] + xyz[2];
    if(children_[child_id] == nullptr) {
      Eigen::Vector3f next_origin;
      next_origin(0) = origin_(0) + xyz[0] * next_voxel_size(0);
      next_origin(1) = origin_(1) + xyz[1] * next_voxel_size(1);
      next_origin(2) = origin_(2) + xyz[2] * next_voxel_size(2);
      children_[child_id] = OctoVoxel::Ptr(new OctoVoxel(next_level, next_voxel_size, next_origin, sem_type_));
    }
    else {
      children_[child_id]->insertPoint(cloud_.points[i]);   
      children_[child_id]->is_updated_ = true;
    }
  }
  cloud_.clear();

  for(int id = 0; id < 8; ++id) {
    if(children_[id] != nullptr)
      children_[id]->parse(thres);
  }
}


void OctoVoxel::getLeafVoxel(std::vector<OctoVoxel>& voxels) {
  if(is_plane_) {
    voxels.push_back(*this);
    return;
  }
  
  for(int id = 0; id < 8; ++id) {
    if(children_[id] != nullptr)
      children_[id]->getLeafVoxel(voxels);
  }
}



void cutCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud, const Transform& Twl, const uint16_t sem_type,
              const Eigen::Vector3f& voxel_size, std::unordered_map<VoxelLoc, OctoVoxel::Ptr>& ovmap) {
  Eigen::Matrix4f Twl_mat = Twl.matrix().cast<float>();
  for (size_t i = 0; i < cloud.size(); i++) {
    const pcl::PointXYZI &point = cloud.points[i];
    pcl::PointXYZI global_point = point;
    global_point.getVector4fMap() = Twl_mat * point.getVector4fMap();
    VoxelLoc position = getVoxelLoc(global_point, voxel_size);
    std::unordered_map<VoxelLoc, OctoVoxel::Ptr>::iterator iter = ovmap.find(position);
    if(iter != ovmap.end()) {
      iter->second->insertPoint(global_point);
    }
    else {
      Eigen::Vector3f origin(position.x * voxel_size.x(), position.y * voxel_size.y(), position.z * voxel_size.z());
      OctoVoxel::Ptr voxel = OctoVoxel::Ptr(new OctoVoxel(0, voxel_size, origin, sem_type));
      voxel->insertPoint(global_point);
      ovmap.insert(std::make_pair(position, voxel));
    }
  }
}

} // namespace SLIM
