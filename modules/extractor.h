#ifndef EXTRACTOR_H
#define EXTRACTOR_H

#include <vector>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_cylinder.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include <opencv2/opencv.hpp>

#include "semantic_definition.h"
#include "observation.h"
#include "utility.h"
#include "voxel.h"
#include "dbscan/surface_cluster.h"
#include "travel/aos.hpp"
#include "travel/tgs.hpp"

namespace SLIM
{
  struct EIGEN_ALIGN16 VelodynePoint
  {
    PCL_ADD_POINT4D;
    float intensity;
    float time;
    uint16_t ring;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };
} // namespace SLIM

POINT_CLOUD_REGISTER_POINT_STRUCT(SLIM::VelodynePoint,
                                  (float, x, x)(float, y, y)(float, z, z)(float, intensity, intensity)(float, time, time)(uint16_t, ring, ring))


struct OusterPoint
{
    PCL_ADD_POINT4D;
    PCL_ADD_INTENSITY;
    uint32_t t;
    uint16_t reflectivity;
    uint16_t ring;
    uint16_t ambient;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (OusterPoint,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity) 
    (uint16_t, ring, ring) (uint16_t, ambient, ambient)
)


namespace SLIM
{

  enum PointAttribute
  {
    NONE = 0,
    EDGE_LEFT = 1,
    EDGE_RIGHT = 2,
    EDGE = 3,
    SURFACE = 4,
    // GROUND = 5,
    // BUILDING = 6,
    POLE = 7,
    // OTHER = 8,
  };

  class ScanPoint
  {
  public:
    ScanPoint()
    {
      pos.setZero();
      curvature = -1;
      range = 1e8;
      null = true;
      yaw = 0;
      attribute = NONE;
    }

    ScanPoint(const Eigen::Vector3f &_pos, const float _yaw)
    {
      pos = _pos;
      range = pos.norm();
      yaw = _yaw;
      null = false;
    }

    void operator=(const ScanPoint &other)
    {
      curvature = other.curvature;
      grad = other.grad;
      yaw = other.yaw;
      range = other.range;
      null = other.null;
      pos = other.pos;
      attribute = other.attribute;
    }

    float curvature = 0;
    float grad = 0;
    float yaw = 0;
    float range = 1e8;
    bool null = true;
    Eigen::Vector3f pos;
    PointAttribute attribute = NONE;
  };

template <typename PointT>
inline int getQuadrant(PointT pt_in) {
  int quadrant = 0;
  double x = pt_in.x;
  double y = pt_in.y;
  if(x > 0 && y >= 0) {
    quadrant = 1;
  } else if(x <= 0 && y > 0) {
    quadrant = 2;
  } else if(x < 0 && y <= 0) {
    quadrant = 3;
  } else {
    quadrant = 4;
  }
  return quadrant;
}

void solveLine(const pcl::PointCloud<pcl::PointXYZI>& cloud, Eigen::Vector3f& mu, Eigen::Vector3f& normal, Eigen::Vector3f& lambda, Eigen::Matrix3f& sigma);

class VLPExtractor
{
public:
  VLPExtractor() = default;

  ~VLPExtractor() = default;

  void setResolution(const int w, const int h);

  void extract(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

  void extract(const std::vector<std::vector<ScanPoint>>& img);

  void extract(const pcl::PointCloud<OusterPoint>::Ptr cloud);

  void extractPole();

  void extractSurface();

  void extractPole(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int scan_line = 64);

  void extractSurface(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int scan_line = 64);

  void refineSurface(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters, const uint16_t sem_type);

  void constructOb();

  pcl::PointCloud<pcl::PointXYZI>::Ptr getEdgeCloud();

  pcl::PointCloud<pcl::PointXYZI>::Ptr getSurfCloud();

  pcl::PointCloud<pcl::PointXYZI>::Ptr getRawSurfCloud();

public:
  std::vector<std::vector<ScanPoint>> img_;
  int w_, h_;
  float fov_up_, fov_down_;
  const float kNumCurvSize = 2;

  pcl::PointCloud<pcl::PointXYZI> raw_surf_cloud_;
  pcl::PointCloud<pcl::PointXYZI> surf_cloud_;
  pcl::PointCloud<pcl::PointXYZI> pole_cloud_;
  std::vector<OctoVoxel> surf_voxels_;
  std::unordered_map<VoxelLoc, OctoVoxel::Ptr> ovmap_;

  std::vector<std::vector<OctoVoxel::Ptr>> cluster_surf_;

  std::vector<LineOB::Ptr> vec_line_feature_;
  std::vector<SurfaceOB::Ptr> vec_surf_feature_;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> line_clusters_;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> road_clusters_;
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> struct_clusters_;
  Transform Twb_;
};

} // namespace SLIM

#endif
