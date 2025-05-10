#ifndef VIEWER_H
#define VIEWER_H

#include <chrono>
#include <thread>
#include <mutex>

#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>  
#include <Eigen/Core>
#include <Eigen/Dense>

#include <pangolin/pangolin.h>
#include <pangolin/gl/glvbo.h>
#include <pangolin/gl/gl.h>
#include <pangolin/gl/gldraw.h>

#include "frame.h"
#include "vector_map.h"
#include "observation.h"
#include "scene_graph.h"

namespace SLIM {

class PangoCloud
{
public:
  typedef std::shared_ptr<PangoCloud> Ptr;
  
  PangoCloud(pcl::PointCloud<pcl::PointXYZI> * cloud)
    : num_points(cloud->size()),
      offset(4),
      stride(sizeof(pcl::PointXYZI)) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
  

  PangoCloud(pcl::PointCloud<pcl::PointXYZRGB> * cloud)
    : num_points(cloud->size()),
      offset(4),
      stride(sizeof(pcl::PointXYZRGB)) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  PangoCloud(pcl::PointCloud<pcl::PointXYZRGBA> * cloud)
    : num_points(cloud->size()),
      offset(4),
      stride(sizeof(pcl::PointXYZRGBA)) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  PangoCloud(pcl::PointCloud<pcl::PointXYZRGBNormal> * cloud)
    : num_points(cloud->size()),
      offset(8),
      stride(sizeof(pcl::PointXYZRGBNormal)) {
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, cloud->points.size() * stride, cloud->points.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  virtual ~PangoCloud() {
    glDeleteBuffers(1, &vbo);
  }

  void drawPoints() {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glVertexPointer(3, GL_FLOAT, stride, 0);
    glColorPointer(4, GL_UNSIGNED_BYTE, stride, (void *)(sizeof(float) * offset)); // 改为4个分量

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // 设置混合函数

    glDrawArrays(GL_POINTS, 0, num_points);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glDisable(GL_BLEND);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  const int num_points;

private:
  const int offset;
  const int stride;
  GLuint vbo;

};

class Viewer {
public:
  typedef std::shared_ptr<Viewer> Ptr;

  Viewer() = default;

  ~Viewer() = default;

  void Run();

  pangolin::OpenGlMatrix GetPGLCameraPose();

  void AddCamera(const Eigen::Matrix4d& Twc, float r, float g, float b, float lw, float scale = 1.0f);

  void DrawCamera(const Eigen::Matrix3d& Rwc, const Eigen::Vector3d& twc);

  void DrawAxis(const Transform& transform, const double scale = 0.1);

  void DrawVectorMap(const VectorMap::Ptr map, const float lw, const bool color_flag, const bool show_connection, const bool show_obs, const uint32_t id=0);

  void DrawTrajectory(const std::map<uint64_t, Frame::Ptr>& traj, const uint32_t id);

  void DrawSemanticGraph();

  void DrawCurFrame(const float width);

  void DrawGraphMatching();

  void DrawMapMatching();

  void DrawLoopCandidate();

  void DrawLoop();

  void DrawSemGraph(const SceneGraph::Ptr graph, int level, bool show_triplet = false);

  void DrawSemMatch(const SceneGraph::Ptr graph_a, const SceneGraph::Ptr graph_b, const std::vector<std::pair<uint32_t, uint32_t>>& matches);

  void DrawLine(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                const double lw, const cv::Scalar color);

  void DrawTriangle(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                  const Eigen::Vector3d& p3, const double lw, const cv::Scalar color);

  void DrawDiamond(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2,
                  const Eigen::Vector3d& p3, const Eigen::Vector3d& p4, 
                  const double lw, const cv::Scalar color);

  void RemoveAllMatchInfo() {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    match_buffer_.clear();
  }

  void SetSemGraphPair(const SceneGraph::Ptr ga, const SceneGraph::Ptr gb) {
    std::unique_lock<std::mutex> lock(graph_matching_mutex_);
    graph_a_ = ga;
    graph_b_ = gb;
  }

  void SetSemGraphMatch(const std::vector<std::pair<uint32_t, uint32_t>>& graph_match) {
    std::unique_lock<std::mutex> lock(graph_matching_mutex_);
    graph_match_ = graph_match;
  }

  void RemoveLoopEdge() {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    loop_info_.clear();
  }

  void AddLoopEdge(const std::pair<Frame::Ptr, Frame::Ptr>& loop) {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    loop_info_.push_back(loop);
  }

  void RemoveOutlierLoopEdge() {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    outlier_loop_info_.clear();
  }

  void AddOutlierLoopEdge(const std::pair<Frame::Ptr, Frame::Ptr>& loop) {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    outlier_loop_info_.push_back(loop);
  }

  void SetPGO(const std::map<uint32_t, Transform>& traj_i, const std::map<uint32_t, Transform>& traj_j, 
              const std::vector<std::pair<uint32_t, uint32_t>>& odom_edge_i, 
              const std::vector<std::pair<uint32_t, uint32_t>>& odom_edge_j, 
              const std::vector<std::pair<uint32_t, uint32_t>>& loop_edge) {
    traj_i_ = traj_i;
    traj_j_ = traj_j;
    odom_edges_i_ = odom_edge_i;
    odom_edges_j_ = odom_edge_j;
    loop_edges_ = loop_edge;
  }

  void AddVectorMap(const VectorMap::Ptr map) {
    vec_map_.push_back(map);
  }

  void SetCurrentPose(const Transform& T) {
    std::unique_lock<std::mutex> lock(frame_mutex_);
    Rwc_ = T.dcm();
    twc_ = T.p();
    pose_ok_ = true;
  }

  void InsertMap(const VectorMap::Ptr map) {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    vec_map_.push_back(map);
  }

  void ClearMapBuffer() {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    vec_map_.clear();
  }

  void SetRefMap(const VectorMap::Ptr map) {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    ref_map_ = map;
  }

  void SetCurMap(const VectorMap::Ptr map) {
    std::unique_lock<std::mutex> lock(feature_mutex_);
    cur_map_ = map;
  }

  void SetStopFlag() {
    std::unique_lock<std::mutex> lock(stop_mutex_);
    stop_ = true;
  }

public:
  Eigen::Matrix3d Rwc_;
  Eigen::Vector3d twc_;
  std::vector<Eigen::Vector3d> trajectory;

  bool pose_ok_ = false;
  bool stop_ = false;
  bool update_ = false;
  bool rendering_ = false;

  std::vector<VectorMap::Ptr> vec_map_;
  VectorMap::Ptr ref_map_, cur_map_;

  SceneGraph::Ptr graph_a_, graph_b_;
  std::vector<std::pair<uint32_t, uint32_t>> graph_match_;

  std::vector<std::pair<uint32_t, uint32_t>> block_matches_;
  std::map<uint32_t, Transform> traj_i_, traj_j_;
  std::vector<std::pair<uint32_t, uint32_t>> odom_edges_i_, odom_edges_j_;
  std::vector<std::pair<uint32_t, uint32_t>> loop_edges_;
  std::vector<std::pair<Frame::Ptr, Frame::Ptr>> loop_info_;
  std::vector<std::pair<Frame::Ptr, Frame::Ptr>> outlier_loop_info_;

  std::vector<Block::Ptr> ref_blocks_, cur_blocks_;
  std::vector<Frame::Ptr> frame_buffer_;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> match_buffer_;  

  std::mutex cloud_mutex_;
  std::mutex frame_mutex_;
  std::mutex feature_mutex_;
  std::mutex stop_mutex_;
  std::mutex graph_matching_mutex_;
  std::mutex map_mutex_;
};


} // namespace SLIM


#endif