#ifndef VECTOR_MAP_H
#define VECTOR_MAP_H

#include <vector>
#include <unordered_map>
#include <mutex>
#include <map>
#include <unordered_map>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <ceres/ceres.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "frame.h"
#include "json.hpp"
#include "factor/line_info.h"
#include "factor/surface_info.h"
#include "observation.h"
#include "landmark.h"
#include "nfr_solver.h"
#include "scene_graph_node.h"
#include "dbscan/surface_cluster.h"
#include "factor/local_param.h"
#include "factor/laser_edge_factor.h"
#include "factor/laser_surf_factor.h"
#include "factor/relative_pose_factor.h"
#include "factor/prior_pose_factor.h"
#include "factor/math_utility.h"
#include "transform.h"

namespace SLIM {

class SceneGraph;

using FrameHashMap = std::unordered_map<uint32_t, Frame::Ptr>;
using LineHashMap = std::unordered_map<uint32_t, LineLM::Ptr>;
using SurfaceHashMap = std::unordered_map<uint32_t, SurfaceLM::Ptr>;
using CentroidCloud = pcl::PointCloud<pcl::PointXYZL>;
using CentroidCloudPtr = pcl::PointCloud<pcl::PointXYZL>::Ptr;
using CentroidKdTree = pcl::search::KdTree<pcl::PointXYZL>;
using CentroidKdTreePtr = pcl::search::KdTree<pcl::PointXYZL>::Ptr;
using NodeQuery = std::pair<pcl::PointCloud<pcl::PointXYZL>::Ptr, pcl::search::KdTree<pcl::PointXYZL>::Ptr>;
using LineMatchVec = std::vector<std::pair<LineLM::Ptr, LineLM::Ptr>>;
using SurfMatchVec = std::vector<std::pair<SurfaceLM::Ptr, SurfaceLM::Ptr>>;

class VectorMap {
  public:
  typedef std::shared_ptr<VectorMap> Ptr;
  VectorMap() {
    id_ = factory_id_++;
    initQuery();
  }

  ~VectorMap() = default;

  void initQuery()
  {
    sem_line_ct_struct_[POLE_ID].first.reset(new CentroidCloud);
    sem_surf_ct_struct_[ROAD_ID].first.reset(new CentroidCloud);
    sem_surf_ct_struct_[SIDEWALK_ID].first.reset(new CentroidCloud);
    sem_surf_ct_struct_[BUILDING_ID].first.reset(new CentroidCloud);
    sem_surf_ct_struct_[FENCE_ID].first.reset(new CentroidCloud);

    sem_line_ct_struct_[POLE_ID].second.reset(new CentroidKdTree);
    sem_surf_ct_struct_[ROAD_ID].second.reset(new CentroidKdTree);
    sem_surf_ct_struct_[SIDEWALK_ID].second.reset(new CentroidKdTree);
    sem_surf_ct_struct_[BUILDING_ID].second.reset(new CentroidKdTree);
    sem_surf_ct_struct_[FENCE_ID].second.reset(new CentroidKdTree);
  }

  bool alignFrame(const Frame::Ptr &frame);

  void addOdomInfo(const Frame::Ptr &frame, const Eigen::Matrix<double, 6, 6>& sqrt_info);

  void addOdomInfo(const RelPoseInfo& info);

  void releaseAll();

  void saveJsonFile(const std::string &file);

  void loadJsonFile(const std::string &file);

  void loadLocalizationMap(const std::string &file);

  void transform(const Transform &Tsd);

  void resolveLM();

  void resort();

  void divideBlocks(const std::map<uint32_t, Frame::Ptr> &pivots, const float range = 50.0f);

  void divideBlocks(const float density = 25.0f, const float range = 50.0f);

  void extractPivots(const float size = 40.0f);

  void findMatchPivots(const VectorMap::Ptr &other, std::vector<std::pair<uint32_t, uint32_t>> &matches);

  void mergePivots(const std::map<uint32_t, Frame::Ptr> &other_pivots);

  void refineRelativePose(VectorMap::Ptr &other,
                          Transform &T,
                          std::map<uint16_t, LineMatchVec> &line_matches,
                          std::map<uint16_t, SurfMatchVec> &surf_matches);

  void visualize(std::map<uint32_t, Transform>& poses, std::map<uint32_t, Transform>& gt_poses, std::vector<std::pair<uint32_t, uint32_t>>& rel_pose_infos,
      std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& poles, std::vector<std::vector<Eigen::Vector3d>>& roads, std::vector<std::vector<Eigen::Vector3d>>& buildings,
      std::vector<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>>& pole_obs, std::vector<std::vector<std::vector<Eigen::Vector3d>>>& road_obs, std::vector<std::vector<std::vector<Eigen::Vector3d>>>& building_obs);

  void filter(const int threshold = 3);

  void prune(const float ratio = 1.0);

  void extractSparseStruct();

  void checkSelf();

  void optimizeFullBA(const double weight = 10.0);

  void evaluateError();

  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> GetMatchInfo();

  size_t FrameSize() const
  {
    return keyframes_.size();
  }

  size_t PoleSize()
  {
    return sem_line_lms_[POLE_ID].size();
  }

  size_t TrafficSignSize()
  {
    return sem_surf_lms_[TRAFFIC_SIGN_ID].size();
  }

  FrameHashMap &getKeyFrames()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return keyframes_;
  }

  std::map<uint16_t, LineHashMap> &getSemLines()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_line_lms_;
  }

  uint32_t getLineLMSize() {
    uint32_t size = 0;
    for(auto cls_iter: sem_line_lms_) {
      size += cls_iter.second.size();
    }
    return size;
  }

  uint32_t getSurfLMSize() {
    uint32_t size = 0;
    for(auto cls_iter: sem_surf_lms_) {
      size += cls_iter.second.size();
    }
    return size;
  }

  std::map<uint16_t, SurfaceHashMap> &getSemSurfaces()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_surf_lms_;
  }

  std::map<uint16_t, NodeQuery> &getSemLineStruct()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_line_ct_struct_;
  }

  std::map<uint16_t, NodeQuery> &getSemSurfStruct()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_surf_ct_struct_;
  }

  std::vector<RelPoseInfo> getOdomInfo() const {
    return odom_info_;
  }

  std::vector<Block::Ptr> getBlocks()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::vector<Block::Ptr> vec_blocks;
    vec_blocks.reserve(blocks_.size());
    for (auto iter : blocks_)
    {
      vec_blocks.push_back(iter.second);
    }
    return vec_blocks;
  }

  std::map<uint32_t, Block::Ptr> &getBlockMap()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return blocks_;
  }

  std::map<uint32_t, Frame::Ptr> &getPivots()
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return pivots_;
  }

  uint32_t getMaxKfId() const {
    uint32_t max_id = 0;
    for(auto &kf_iter: keyframes_) {
      if(kf_iter.second->id() > max_id) {
        max_id = kf_iter.second->id();
      }
    }
    return max_id;
  }

  void insertFrame(const Frame::Ptr &frame)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    keyframes_.insert(std::make_pair(frame->id(), frame));
  }

  void insertLine(const LineLM::Ptr &lm)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    sem_line_lms_[lm->semantic_type()].insert(std::make_pair(lm->id(), lm));
  }

  void insertSurface(const SurfaceLM::Ptr &lm)
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    sem_surf_lms_[lm->semantic_type()].insert(std::make_pair(lm->id(), lm));
  }

  void saveTumTrajectory(const std::string &filename);

  void saveTrajAndCloudPath(const std::string &filename);

  void saveSyncBenchmark(const std::string &filename, const std::map<uint64_t, Transform> &gt);

  void saveSummary(const std::string& summary_file);

  void initStruct();

  void printInfo();

 public:
  void insertNewLine(const Frame::Ptr &frame, const LineOB::Ptr &ob);
  void insertNewSurface(const Frame::Ptr &frame, const SurfaceOB::Ptr &ob);
  void AssociateLine(const Frame::Ptr &frame);
  void AssociateSurface(const Frame::Ptr &frame);
  SurfaceLM::Ptr getLargestSurf(const std::vector<SurfaceLM::Ptr>& surfs);
  std::vector<SurfaceLM::Ptr> extractConsistentSurf(const SurfaceLM::Ptr& ref, const std::vector<SurfaceLM::Ptr>& surfs);

  Json::Value writeFrameObject(const Frame::Ptr frame);
  Json::Value writeLineObject(const LineLM::Ptr landmark);
  Json::Value writeSurfObject(const SurfaceLM::Ptr landmark);
  Json::Value writeOdomInfo();

  Frame::Ptr loadFrame(const Json::Value &f);
  LineLM::Ptr loadLineObject(const Json::Value &f, const bool loc = false);
  SurfaceLM::Ptr loadSurfObject(const Json::Value &f, const bool loc = false);
  void loadOdomInfo(const Json::Value &root, const bool loc = false);

  static uint32_t factory_id_;
  uint32_t id_;

  FrameHashMap keyframes_;
  std::map<uint16_t, LineHashMap> sem_line_lms_;
  std::map<uint16_t, SurfaceHashMap> sem_surf_lms_;
  std::map<uint16_t, NodeQuery> sem_line_ct_struct_;
  std::map<uint16_t, NodeQuery> sem_surf_ct_struct_;
  std::vector<RelPoseInfo> odom_info_;
  Frame::Ptr pivot_kf_;
  Eigen::Matrix<double, 6, 6> pivot_sqrt_info_;

  std::map<uint32_t, Block::Ptr> blocks_;
  std::map<uint32_t, Frame::Ptr> pivots_;
  Frame::Ptr last_kf_;
  
  double ba_time_, nfr_time_;

  std::mutex data_mutex_;
};

} // namespace SLIM

#endif