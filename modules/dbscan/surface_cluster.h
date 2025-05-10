#ifndef SURFACE_CLUSTER_H
#define SURFACE_CLUSTER_H

#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include "dbscan/simple_cluster.h"
#include "factor/math_utility.h"

namespace SLIM {

#define UN_PROCESSED 0
#define PROCESSING 1
#define PROCESSED 2

class SurfaceCluster {
 public:
  typedef typename pcl::PointCloud<pcl::PointXYZINormal>::Ptr PointCloudPtr;
  typedef typename pcl::search::KdTree<pcl::PointXYZINormal>::Ptr KdTreePtr;

  SurfaceCluster() = default;

  virtual void setInputCloud(PointCloudPtr cloud) {
    input_cloud_ = cloud;
  }

  void extract(std::vector<pcl::PointIndices>& cluster_indices) {
    std::vector<int> nn_indices;
    std::vector<float> nn_distances;
    std::vector<bool> is_noise(input_cloud_->points.size(), false);
    std::vector<int> types(input_cloud_->points.size(), UN_PROCESSED);
    for (int i = 0; i < input_cloud_->points.size(); i++) {
      if (types[i] == PROCESSED) {
        continue;
      }
      int nn_size = radiusSearch(i, radius_tolerance_, nn_indices, nn_distances);
      if (nn_size < min_point_num_) {
        is_noise[i] = true;
        continue;
      }
      
      std::vector<int> seed_queue;
      seed_queue.push_back(i);
      types[i] = PROCESSED;
      
      for (int j = 0; j < nn_size; j++) {
        if (nn_indices[j] != i) {
          seed_queue.push_back(nn_indices[j]);
          types[nn_indices[j]] = PROCESSING;
        }
      } // for every point near the chosen core point.
      int sq_idx = 1;
      while (sq_idx < seed_queue.size()) {
        int cloud_index = seed_queue[sq_idx];
        if (is_noise[cloud_index] || types[cloud_index] == PROCESSED) {
          // seed_queue.push_back(cloud_index);
          types[cloud_index] = PROCESSED;
          sq_idx++;
          continue; // no need to check neighbors.
        }
        nn_size = radiusSearch(cloud_index, radius_tolerance_, nn_indices, nn_distances);
        if (nn_size >= min_point_num_) {
          for (int j = 0; j < nn_size; j++) {
            if (types[nn_indices[j]] == UN_PROCESSED) {   
              seed_queue.push_back(nn_indices[j]);
              types[nn_indices[j]] = PROCESSING;
            }
          }
        }
        
        types[cloud_index] = PROCESSED;
        sq_idx++;
      }
      if (seed_queue.size() >= min_pts_per_cluster_ && seed_queue.size () <= max_pts_per_cluster_) {
        pcl::PointIndices r;
        r.indices.resize(seed_queue.size());
        for (int j = 0; j < seed_queue.size(); ++j) {
          r.indices[j] = seed_queue[j];
        }
        // These two lines should not be needed: (can anyone confirm?) -FF
        std::sort(r.indices.begin(), r.indices.end());
        r.indices.erase(std::unique(r.indices.begin(), r.indices.end()), r.indices.end());

        r.header = input_cloud_->header;
        cluster_indices.push_back(r);   // We could avoid a copy by working directly in the vector
      }
    } // for every point in input cloud
    std::sort(cluster_indices.rbegin(), cluster_indices.rend(), comparePointClusters);
  }

  void setClusterTolerance(double tolerance) {
    radius_tolerance_ = tolerance; 
  }

  void setAngleTolerance(double tolerance) {
    angle_tolerance_ = tolerance;
  }

  void setDistTolerance(double tolerance) {
    dist_tolerance_ = tolerance;
  }

  void setMinClusterSize (int min_cluster_size) { 
    min_pts_per_cluster_ = min_cluster_size; 
  }

  void setMaxClusterSize (int max_cluster_size) { 
    max_pts_per_cluster_ = max_cluster_size; 
  }
  
  void setCorePointMinPts(int core_point_min_pts) {
    min_point_num_ = core_point_min_pts;
  }

 protected:
  PointCloudPtr input_cloud_;
  
  double radius_tolerance_ {0.0};
  double angle_tolerance_{M_PI / 36.0};
  double dist_tolerance_{0.3};

  int min_point_num_ {1}; // not including the point itself.
  int min_pts_per_cluster_ {1};
  int max_pts_per_cluster_ {std::numeric_limits<int>::max()};
  

  virtual int radiusSearch(int index, double radius, std::vector<int> &k_indices, std::vector<float> &k_sqr_distances) const {
    k_indices.clear();
    k_sqr_distances.clear();
    k_indices.push_back(index);
    k_sqr_distances.push_back(0);
    int size = input_cloud_->points.size();
    double radius_sq = radius * radius;
    for (int i = 0; i < size; i++) {
      if (i == index) {
        continue;
      }
      pcl::PointXYZINormal const& point_i = input_cloud_->points[i];
      pcl::PointXYZINormal const& point_index = input_cloud_->points[index];
      if(std::round(point_i.intensity) != std::round(point_index.intensity)) {
        continue;
      }
      Eigen::Vector3f const centroid_i = point_i.getVector3fMap();
      Eigen::Vector3f const centroid_index = point_index.getVector3fMap();
      double dist_sq = (centroid_i - centroid_index).norm();
      if(dist_sq > radius) {
        continue;
      }
      Eigen::Vector3f const normal_i{point_i.normal_x, point_i.normal_y, point_i.normal_z};
      Eigen::Vector3f const normal_index{point_index.normal_x, point_index.normal_y, point_index.normal_z};
      double theta = std::acos(normal_i.dot(normal_index));
      double dist = std::abs(normal_index.dot(centroid_i - centroid_index));
      
      if((theta < angle_tolerance_ || theta > M_PI - angle_tolerance_) && dist < dist_tolerance_) {
        k_indices.push_back(i);
        k_sqr_distances.push_back(std::sqrt(dist_sq * dist_sq));
      }
    }
    return k_indices.size();
  }
}; // class SurfaceCluster
} // namespace SLIM

#endif // SIMPLE_CLUSTER_H