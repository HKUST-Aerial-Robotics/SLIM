#ifndef VOXEL_H
#define VOXEL_H

#include <vector>
#include <cstdlib>
#include <algorithm>
#include <stdio.h>
#include <unordered_map>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>

#include "transform.h"
#include "semantic_definition.h"

namespace SLIM {
  
class VoxelLoc {
 public:
  int64_t x, y, z;
  VoxelLoc(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VoxelLoc &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

} // namespace SLIM

// Hash value
namespace std {
using namespace SLIM;
  template<>
  struct hash<VoxelLoc> {
    size_t operator() (const VoxelLoc &s) const {
      using std::size_t; using std::hash;
      return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
    }
  };
}

namespace SLIM {

class Voxel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<Voxel> Ptr;
  Voxel(const Eigen::Vector3f& voxel_size, const Eigen::Vector3d& origin, const uint16_t semantic_type)
  : voxel_size_(voxel_size), origin_(origin), semantic_type_(semantic_type) {
    center_.setZero();
    normal_.setZero();
    sigma_.setZero();
  }

  Eigen::Vector3d voxel_center() const { 
    return origin_ + voxel_size_.cast<double>() * 0.5; 
  }

  const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud() const {
    return cloud_.makeShared();
  }

  const Eigen::Vector3d& center() {
    return center_;
  }

  const Eigen::Vector3d& normal() {
    return normal_;
  }

  const Eigen::Matrix3d& sigma() {
    return sigma_;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud() {
    return cloud_.makeShared();
  }

  void insertPoint(const pcl::PointXYZI& point) {
    cloud_.push_back(point);
  }

  bool parse(const int mode);

 private:
  Eigen::Vector3d origin_;
  Eigen::Vector3f voxel_size_;
  Eigen::Vector3d center_, normal_, lambda_;
  Eigen::Matrix3d sigma_;
  pcl::PointCloud<pcl::PointXYZI> cloud_;
  uint16_t semantic_type_;
  bool is_updated_;
};

VoxelLoc getVoxelLoc(const pcl::PointXYZI& point, const Eigen::Vector3f& voxel_size);

void cutCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud, const Transform& Twl, const uint16_t semantic_type,
              const Eigen::Vector3f& voxel_size, std::unordered_map<VoxelLoc, Voxel::Ptr>& voxel_map);

template <typename PointT>
inline void solveCovMat(const pcl::PointCloud<PointT>& cloud, Eigen::Vector3f& mu, Eigen::Matrix3f& cov) {
  mu.setZero();
  cov.setZero();
  Eigen::Vector3f point;
  auto N = cloud.size();
  for(int i = 0; i < N; ++i) {
    point = cloud.points[i].getVector3fMap();
    mu += point;
    cov += point * point.transpose();
  } 
  mu /= N;
  cov.noalias() = cov / N - mu * mu.transpose();
}

class OctoVoxel {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef std::shared_ptr<OctoVoxel> Ptr;
  OctoVoxel(int level, const Eigen::Vector3f& voxel_size, const Eigen::Vector3f& origin, const uint16_t sem_type)
  : level_(level), voxel_size_(voxel_size), origin_(origin), sem_type_(sem_type), is_leaf_(true), is_updated_(true), is_plane_(false),
    center_(Eigen::Vector3f::Zero()), normal_(Eigen::Vector3f::Zero()), sigma_(Eigen::Matrix3f::Zero()) {
    is_root_ = level == 0 ? true : false;
    cloud_.clear();
    Eigen::Vector3f point_color = Eigen::Vector3f::Random();
    red = std::abs(point_color(0)) / 1.0f * 255;
    green = std::abs(point_color(1)) / 1.0f * 255;
    blue = std::abs(point_color(2)) / 1.0f * 255;
  }

  bool checkPCA(const double thres = 0.2);

  bool hasLeafNode() {
    if(is_leaf_) {
      return true;
    }
    
    for(int id = 0; id < 8; ++id) {
      if(children_[id] == nullptr) continue;
      if(children_[id]->is_leaf_) {
        return true;
      }
      else {
        if(children_[id]->hasLeafNode()) {
          return false;
        }
      }
    }
    return false;
  }

  void parse(const double thres = 0.2);

  int getChildrenNum() {
    int num = 0;
    for(int id = 0; id < 8; ++id) {
      if(children_[id] != nullptr)
        num++;
    }
    return num;
  }

  void prune(const std::set<OctoVoxel::Ptr>& leaves) {
    if(is_plane_) {
      return;
    }

    for(int id = 0; id < 8; ++id) {
      if(children_[id] == nullptr) {
        continue;
      }
      children_[id]->prune(leaves);
      if(leaves.find(children_[id]) == leaves.end() && children_[id]->getChildrenNum() == 0) {
        children_[id] = nullptr;
      }
    }
  }

  void getVoxelVertices(std::vector<std::vector<Eigen::Vector3f>>& all_vertices) {
    if(is_plane_) {
      std::vector<Eigen::Vector3f> vertices;
      vertices.push_back(origin_);
      vertices.push_back(origin_ + Eigen::Vector3f(voxel_size_.x(), 0., 0.));
      vertices.push_back(origin_ + Eigen::Vector3f(0., voxel_size_.y(), 0.));
      vertices.push_back(origin_ + Eigen::Vector3f(voxel_size_.x(), voxel_size_.y(), 0.));
      vertices.push_back(origin_ + Eigen::Vector3f(0., 0., voxel_size_.z()));
      vertices.push_back(origin_ + Eigen::Vector3f(voxel_size_.x(), 0., voxel_size_.z()));
      vertices.push_back(origin_ + Eigen::Vector3f(0., voxel_size_.y(), voxel_size_.z()));
      vertices.push_back(origin_ + Eigen::Vector3f(voxel_size_.x(), voxel_size_.y(), voxel_size_.z()));
      all_vertices.push_back(vertices);
      return;
    }
    for(int id = 0; id < 8; ++id) {
      if(children_[id] != nullptr)
        children_[id]->getVoxelVertices(all_vertices);
    }
  }

  void getLeafCloud(std::vector<std::pair<OctoVoxel::Ptr, pcl::PointCloud<pcl::PointXYZI>>>& leaf_nodes) {
    if(is_plane_) {
      leaf_nodes.emplace_back(std::make_shared<OctoVoxel>(*this), cloud_);
      return;
    }
    
    for(int id = 0; id < 8; ++id) {
      if(children_[id] != nullptr)
        children_[id]->getLeafCloud(leaf_nodes);
    }
  }

  void getLeafVoxel(std::vector<OctoVoxel>& voxels);

  void downsample() {
    if(is_plane_ || is_leaf_) {
      pcl::VoxelGrid<pcl::PointXYZI> vf;
      vf.setLeafSize(0.2, 0.2, 0.2);
      vf.setInputCloud(cloud_.makeShared());
      vf.filter(cloud_);
      return;
    }
    for(int id = 0; id < 8; ++id) {
      if(children_[id] != nullptr)
        children_[id]->downsample();
    }
  }

  Eigen::Vector3f voxel_center() const { 
    return origin_ + voxel_size_ * 0.5f; 
  }

  const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud() const {
    return cloud_.makeShared();
  }

  const Eigen::Vector3f& center() {
    return center_;
  }

  const Eigen::Vector3f& normal() {
    return normal_;
  }

  const Eigen::Matrix3f& sigma() {
    return sigma_;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud() {
    return cloud_.makeShared();
  }

  void insertPoint(const pcl::PointXYZI& point) {
    cloud_.push_back(point);
    is_updated_ = true;
  }

 public:
  Eigen::Vector3f origin_;
  Eigen::Vector3f voxel_size_;
  Eigen::Vector3f center_, normal_, lambda_;
  Eigen::MatrixX3f sigma_;
  unsigned char red, green, blue;
  pcl::PointCloud<pcl::PointXYZI> cloud_;
  uint16_t sem_type_;
  OctoVoxel::Ptr children_[8];
  int level_;
  bool is_root_ = true;
  bool is_leaf_ = true;
  bool is_updated_ = false;
  bool is_plane_ = false;
  static int max_level_; 
};


void cutCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud, const Transform& Twl, const uint16_t sem_type,
              const Eigen::Vector3f& voxel_size, std::unordered_map<VoxelLoc, OctoVoxel::Ptr>& ovmap);

std::vector<OctoVoxel::Ptr> searchNeighbors(const std::unordered_map<VoxelLoc, OctoVoxel::Ptr>& ovmap, const OctoVoxel::Ptr start);

} // namespace SLIM


#endif