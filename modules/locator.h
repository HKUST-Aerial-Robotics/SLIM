#ifndef LOCATOR_H
#define LOCATOR_H

#include "vector_map.h"
#include "frame.h"
#include "scene_graph.h"
#include "viewer.h"
#include "clipper.h"

namespace SLIM {

  
class Locator {
 public:
  Locator() = default;

  void setMap(const VectorMap::Ptr& map);

  void setViewer(const Viewer::Ptr& viewer);

  void setTrajFileName(const std::string& traj_file_name);

  bool solveRelativePose(const SceneGraph::Ptr& gref, const SceneGraph::Ptr& gcur, 
                         const clipper::Association& matches, Transform& Trc);

  bool refineRelativePose(const Block::Ptr& block, const pcl::PointCloud<pcl::PointXYZI>::Ptr& sem_cloud, Transform& Tij);

  bool solve(uint64_t timestamp, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud);

  void saveTumTrajectory(const std::string &filename);

  bool solve(
    uint64_t timestamp, 
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pole, 
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& road,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& building
  );

//  private:
  VectorMap::Ptr map_;
  Viewer::Ptr viewer_;
  std::vector<Block::Ptr> blocks_;
  std::vector<SceneGraph::Ptr> graphs_;

  std::vector<std::pair<double, Transform>> vec_reloc_info_;
  double cur_line_ratio_;
  double cur_surf_ratio_;

  Transform cur_Twb_;
  Transform last_Twb_;
  Transform delta_Twb_;
  double last_time_;
  bool initialized_ = false;

  std::string traj_file_name_;
  std::map<uint64_t, Transform> trajectory_;

};

} // namespace SLIM

#endif