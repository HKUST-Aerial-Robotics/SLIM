#ifndef FRAME_H
#define FRAME_H

#include <vector>
#include <memory>
#include <mutex>

#include <Eigen/Core>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>

#include <pcl/segmentation/extract_clusters.h>

#include "observation.h"
#include "transform.h"
#include "voxel.h"
#include "utility.h"
#include "dbscan/kdtree_cluster.h"
#include "dbscan/surface_cluster.h"

namespace SLIM {

class Frame {
 public:
  typedef std::shared_ptr<Frame> Ptr;
  Frame() = default;

  Frame(const uint64_t timestamp, const Transform& Twb);

  Frame(const uint32_t id, const uint64_t timestamp, const Transform& Twb);

  Frame(const uint32_t id, const uint64_t timestamp, const Transform& Twb, 
        const std::vector<LineOB::Ptr>& line_obs,
        const std::vector<SurfaceOB::Ptr>& surf_obs);

  ~Frame() = default;

  void DownSample(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const float resolution);

  void setCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud) {
    cloud_ = cloud;
  }

  void SetSemanticColorCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud) {
    sem_color_cloud_ = cloud;
  }

  pcl::PointCloud<pcl::PointXYZI>::Ptr getGlobalCloud() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    pcl::PointCloud<pcl::PointXYZI>::Ptr global_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    Eigen::Matrix4f Tf = Twb_.matrix().cast<float>();
    for(int i = 0; i < cloud_->size(); ++i) {
      pcl::PointXYZI point(cloud_->points[i]);
      point.getVector4fMap() = Tf * cloud_->points[i].getVector4fMap();
      global_cloud->push_back(point);
    }
    return global_cloud;
  }

  void setTwb(const Transform& Twb) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    Twb_ = Twb;
  }

  void setID(const uint32_t id) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    id_ = id;
  }

  void setTime(const uint64_t time) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    timestamp_ = time;
  }

  uint32_t id() const {
    return id_;
  }

  uint64_t timestamp() const {
    return timestamp_;
  }

  Transform& Twb() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return Twb_;
  }

  Transform const constTwb() const {
    return Twb_;
  }

  Transform Twl() const {
    return Twb_ * Tbl_;
  }

  std::vector<LineOB::Ptr> line_obs() const {
    return line_obs_;
  }

  std::vector<SurfaceOB::Ptr> surface_obs() const {
    return surface_obs_;
  }

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sem_color_cloud() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    return sem_color_cloud_;
  }

 public:
  static uint32_t factory_id_;
  uint32_t id_;
  uint64_t timestamp_;
  Transform Twb_;
  Transform gt_Twb_;
  Transform Tbl_;
  std::vector<LineOB::Ptr> line_obs_;
  std::vector<SurfaceOB::Ptr> surface_obs_;

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr sem_color_cloud_;
  std::string cloud_path_;

  std::mutex data_mutex_;
};
} // namespace SLIM

#endif