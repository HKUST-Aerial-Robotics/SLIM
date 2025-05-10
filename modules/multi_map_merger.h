#ifndef MULTI_MAP_MERGER_H
#define MULTI_MAP_MERGER_H

#include "vector_map.h"
#include "scene_graph.h"
#include "viewer.h"
#include "pcm_solver.h"
#include "clipper.h"
#include "utility.h"
#include "nfr_solver.h"
#include <iostream>
#include <cassert>

#include <unordered_set>
#include <unordered_map>

namespace SLIM {



using Graph = std::unordered_map<int, std::unordered_set<int>>;

std::unordered_set<int> findConnectedComponent(const Graph& graph, int startNode);

std::vector<int> findConnectedComponents(const Graph& graph);


struct ExpInfo {
  double nfr_time;
  double optimize_time;
  double map_size; // MB
  uint32_t frame_size, line_size, surf_size;
  
};

struct LoopInfo {
  VectorMap::Ptr mi, mj;
  RelPoseInfo rel_pose_info;
  LoopInfo(const VectorMap::Ptr _mi, const VectorMap::Ptr _mj, const RelPoseInfo& _rel_pose_info): mi(_mi), mj(_mj), rel_pose_info(_rel_pose_info) {}
};

struct MapNode {
  typedef std::shared_ptr<MapNode> Ptr;
  int id;
  std::vector<MapNode::Ptr> neighbors;
  VectorMap::Ptr map;
  bool visited;
  MapNode(int _id) : id(_id), map(nullptr), visited(false) {}
  MapNode(int _id, VectorMap::Ptr _map) : id(_id), map(_map), visited(false) {}
};

// DFS遍历
void DFS(MapNode::Ptr node, std::unordered_set<int>& connectedSet);

std::vector<int> findMaxConnectedGraph(std::vector<MapNode::Ptr>& graph);

class MultiMapMerger {
 public:
  MultiMapMerger() {
    
  }

  void setViewer(const Viewer::Ptr viewer) {
    viewer_ = viewer;
  }

  void addSubMap(const VectorMap::Ptr submap) {
    map_cache_.push_back(submap);
  }

  void findLoopMS();

  Block::Ptr initBlock(const Frame::Ptr frame, const VectorMap::Ptr map, const float range = 40.0f);

  bool findPairLoop(const VectorMap::Ptr& mi, const VectorMap::Ptr& mj, 
      const std::vector<Block::Ptr>& blocks0, const std::vector<Block::Ptr>& blocks1,
      const std::vector<SceneGraph::Ptr>& graphs0, const std::vector<SceneGraph::Ptr>& graphs1, std::vector<OverlapInfo>& ov_infos);

  void selectLoop(std::vector<LoopInfo>& loop_info, std::vector<OverlapInfo>& ov_info);

  void alignMultiMap();

  void setRefMap(const VectorMap::Ptr ref_map) {
    ref_map_ = ref_map;
    // std::map<uint32_t, Frame::Ptr> trajectory;
    // for(auto &f_iter: ref_map_->getKeyFrames()) {
    //   Transform& Twb = f_iter.second->Twb();
    //   trajectory.insert(std::make_pair(f_iter.second->id(), f_iter.second));
    // }
    // std::vector<RelPoseInfo> odom_info;
    // for(auto iter = trajectory.begin(); iter != std::prev(trajectory.end()); ++iter) {
    //   Frame::Ptr const fi = iter->second;
    //   Frame::Ptr const fj = std::next(iter)->second;
    //   Transform const Tij = fi->Twb().inverse() * fj->Twb();
    //   odom_info.push_back(RelPoseInfo(fi, fj, Tij));
    // }
    // vec_odom_info_.push_back(odom_info);
    // merge_num_++;

    // saveTumTrajectory(output_path_ + "/merge_num_" + std::to_string(merge_num_) + "_ori.txt");
  }

  void setCurMap(const VectorMap::Ptr cur_map) {
    cur_map_ = cur_map;
    // std::map<uint32_t, Frame::Ptr> trajectory;
    // for(auto &f_iter: cur_map_->getKeyFrames()) {
    //   Transform& Twb = f_iter.second->Twb();
    //   trajectory.insert(std::make_pair(f_iter.second->id(), f_iter.second));
    // }
    // cur_odom_info_.clear();
    // for(auto iter = trajectory.begin(); iter != std::prev(trajectory.end()); ++iter) {
    //   Frame::Ptr const fi = iter->second;
    //   Frame::Ptr const fj = std::next(iter)->second;
    //   Transform const Tij = fi->Twb().inverse() * fj->Twb();
    //   cur_odom_info_.push_back(RelPoseInfo(fi, fj, Tij));
    // }
    // merge_num_++;
  }

  void setOutputPath(const std::string& output_path) {
    output_path_ = output_path;
    createDirectory(output_path_ + "/pgo");
    createDirectory(output_path_ + "/ba");
    createDirectory(output_path_ + "/submap");
    createDirectory(output_path_ + "/summary");
  }

  bool solveRelativePose(const SceneGraph::Ptr& gref, const SceneGraph::Ptr& gcur, 
                         const clipper::Association& matches, Transform& Trc);

  bool refineRelativePose(const Block::Ptr& block_i, const Block::Ptr& block_j, Transform& Tij, Eigen::Matrix<double, 6, 6>& sqrt_info);

  bool refineRelativePoseGNC(const Block::Ptr& block_i, const Block::Ptr& block_j, Transform& Tij);

  std::vector<int> solvePCM(const std::vector<OverlapInfo>& info);

  std::vector<OverlapInfo> searchPotentialOverlap(const std::vector<Block::Ptr>& ref_blocks, 
                                                  const std::vector<Block::Ptr>& cur_blocks,
                                                  const std::set<Block::Ptr>& ref_set,
                                                  const std::set<Block::Ptr>& cur_set,
                                                  const Transform& Trc);

  void optimizePGO(const VectorMap::Ptr& ref_map, const VectorMap::Ptr& cur_map, const std::vector<OverlapInfo>& overlap);

  void optimizePGO(const std::vector<OverlapInfo>& overlap);

  void optimizeGncPGO(const std::vector<OverlapInfo>& overlap);

  void combineMap(const VectorMap::Ptr& map);
  
  // void combineMap(VectorMap::Ptr& ref_map, const VectorMap::Ptr& cur_map);

  void combineMap();

  void optimizeGBA();

  bool merge();

  void saveTumTrajectory(const std::string& filename);

  Transform solveAveragePose(const std::vector<OverlapInfo>& ov_info);

  VectorMap::Ptr getRefMap() {
    return ref_map_;
  }

  VectorMap::Ptr getCurMap() {
    return cur_map_;
  }
  
 public:
  VectorMap::Ptr ref_map_, cur_map_;
  VectorMap::Ptr base_map_;
  Viewer::Ptr viewer_;

  std::vector<VectorMap::Ptr> map_cache_;
  std::vector<VectorMap::Ptr> map_queue_;

  std::vector<OverlapInfo> ov_info_;
  std::vector<std::pair<Frame::Ptr, std::pair<Frame::Ptr, Transform>>> vec_rel_pose_;
  std::string output_path_;

  std::vector<std::vector<RelPoseInfo>> vec_odom_info_;
  std::vector<std::vector<RelPoseInfo>> vec_loop_info_;
  std::vector<RelPoseInfo> cur_odom_info_;
  std::vector<RelPoseInfo> cur_loop_info_;

  std::vector<LoopInfo> loop_info_;
  std::map<VectorMap::Ptr, std::map<VectorMap::Ptr, std::vector<LoopInfo>>> loop_info_map_;
  std::map<VectorMap::Ptr, std::map<VectorMap::Ptr, std::vector<OverlapInfo>>> ov_info_map_;
  std::unordered_map<Frame::Ptr, Block::Ptr> block_map_;
  std::unordered_map<Frame::Ptr, SceneGraph::Ptr> graph_map_;

  std::vector<std::vector<Block::Ptr>> block_vec_;
  std::vector<std::vector<SceneGraph::Ptr>> graph_vec_;
  std::map<uint32_t, ExpInfo> exp_info_;

  uint32_t merge_num_ = 0;
  bool use_ba_;
};

} // namespace SLIM



#endif