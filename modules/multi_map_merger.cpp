#include "multi_map_merger.h"
#include <assert.h>

namespace SLIM {

std::unordered_set<int> findConnectedComponent(const Graph& graph, int startNode) {
  std::unordered_set<int> visited;
  std::queue<int> queue;
  queue.push(startNode);

  while (!queue.empty()) {
    int currentNode = queue.front();
    queue.pop();
    visited.insert(currentNode);

    const auto& neighbors = graph.at(currentNode);
    for (int neighbor : neighbors) {
      if (visited.count(neighbor) == 0) {
        queue.push(neighbor);
      }
    }
  }

  return visited;
}

std::vector<int> findConnectedComponents(const Graph& graph) {
  std::unordered_set<int> visited;
  std::vector<std::unordered_set<int>> components;

  for (const auto& entry : graph) {
    int node = entry.first;
    if (visited.count(node) == 0) {
      std::unordered_set<int> component = findConnectedComponent(graph, node);
      components.push_back(component);
      visited.insert(component.begin(), component.end());
    }
  }

  int max_len = 0;
  int max_id = 0;
  for (int i = 0; i < components.size(); i++) {
    if(components[i].size() > max_len) {
      max_id = i;
    }
  }
  std::vector<int> max_graph = std::vector<int>(components[max_id].begin(), components[max_id].end());
  std::sort(max_graph.begin(), max_graph.end());
  return max_graph;
}


void DFS(MapNode::Ptr node, std::unordered_set<int>& connectedSet) {
  std::stack<MapNode::Ptr> stk;
  stk.push(node);

  while (!stk.empty()) {
    MapNode::Ptr curr = stk.top();
    stk.pop();

    if (curr->visited) {
      continue;
    }

    curr->visited = true;
    connectedSet.insert(curr->id);

    for (MapNode::Ptr neighbor : curr->neighbors) {
      if (!neighbor->visited) {
        stk.push(neighbor);
      }
    }
  }
}

std::vector<int> findMaxConnectedGraph(std::vector<MapNode::Ptr>& graph) {
  std::vector<int> result;
  std::unordered_set<int> connectedSet;

  for (MapNode::Ptr node : graph) {
    if (!node->visited) {
      connectedSet.clear();
      DFS(node, connectedSet);

      if (connectedSet.size() > 1) {
        for (int id : connectedSet) {
          result.push_back(id);
        }
      }
    }
  }

  return result;
}

bool MultiMapMerger::merge() {
  cv::Mat bottom(100, 100, CV_8U, cv::Scalar(255));

  viewer_->SetRefMap(ref_map_);
  viewer_->SetCurMap(cur_map_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);

  printf("[MapMerger][Merge] Start to divide blocks... \n");
  ref_map_->divideBlocks(10.0f, 40.0f);
  std::vector<Block::Ptr> ref_blocks = ref_map_->getBlocks();
  cur_map_->divideBlocks(10.0f, 40.0f);
  std::vector<Block::Ptr> cur_blocks = cur_map_->getBlocks();

  printf("[MapMerger][Merge] Start to build semantic graphs... \n");
  std::vector<SceneGraph::Ptr> ref_graphs, cur_graphs;
  for(auto &block: ref_blocks) {
    SceneGraph::Ptr graph(new SceneGraph);
    graph->buildFromBlock(block);
    graph->buildSelfAffinity();
    ref_graphs.push_back(graph);
  }

  for(auto &block: cur_blocks) {
    SceneGraph::Ptr graph(new SceneGraph);
    graph->buildFromBlock(block);
    graph->buildSelfAffinity();
    cur_graphs.push_back(graph);
  }

  ov_info_.clear();

  Transform Trc;
  std::set<Block::Ptr> ref_set, cur_set;
  bool stop = false;
  std::vector<std::pair<Frame::Ptr, Frame::Ptr>> outlier_loop_info;
  for(int i = 0; i < ref_graphs.size(); ++i) {
    if(stop) break;
    if(ref_set.find(ref_blocks[i]) != ref_set.end())
      continue;
    for(int j = 0; j < cur_graphs.size(); ++j) {
      if(cur_set.find(cur_blocks[j]) != cur_set.end())
        continue;

      printf("[MapMerger] Register Graph! (%d, %d)\n", i, j);
      GlobalRegister global_reg;
      auto association = global_reg.solveMatch(ref_graphs[i], cur_graphs[j]);

      double ref_ratio = (double)association.rows() / ref_graphs[i]->nodes_.size();
      double cur_ratio = (double)association.rows() / cur_graphs[j]->nodes_.size();
      if(ref_ratio < 0.4 || cur_ratio < 0.4) {
        printf("[MapMerger] Too low overlap ratio! (%lf, %lf)\n", ref_ratio, cur_ratio);
        continue;
      }

      Transform Tij;
      Eigen::Matrix<double, 6, 6> sqrt_info;
      bool suc = false;
      if(!solveRelativePose(ref_graphs[i], cur_graphs[j], association, Tij)) {
        printf("[MapMerger] Failed in blocks coarse align!\n");
        // outlier_loop_info.push_back(std::make_pair(bi->getHostFrame(), bj->getHostFrame()));
        continue;
      }

      if(!refineRelativePose(ref_blocks[i], cur_blocks[j], Tij, sqrt_info)) {
      // if(!refineRelativePoseGNC(ref_blocks[i], cur_blocks[j], Tij)) {
        printf("[MapMerger] Failed in blocks refine align!\n");
        // outlier_loop_info.push_back(std::make_pair(ref_blocks[i]->getHostFrame(), cur_blocks[j]->getHostFrame()));
        continue;
      }

      const Transform Twbi = ref_blocks[i]->getHostFrame()->Twb();
      const Transform Twbj = cur_blocks[j]->getHostFrame()->Twb();
      auto first_info = OverlapInfo(ref_blocks[i], cur_blocks[j], ref_map_, cur_map_, Twbi.inverse() * Tij * Twbj, Tij, sqrt_info);
      ov_info_.push_back(first_info);
      ref_set.insert(ref_blocks[i]);
      cur_set.insert(cur_blocks[j]);

      Trc = Tij;
      printf("[MapMerger][Merge] System has found the first overlap and starts to find more potential overlap ...\n");
      std::vector<OverlapInfo> pot_ov_infos = searchPotentialOverlap(ref_blocks, cur_blocks, ref_set, cur_set, Trc);
      ov_info_.insert(ov_info_.end(), pot_ov_infos.begin(), pot_ov_infos.end());

      if(ov_info_.size() < 5) {
        printf("[MapMerger] There are %ld loop frames, reject!\n", ov_info_.size());
        ov_info_.clear();
        ref_set.erase(ref_blocks[i]);
        cur_set.erase(cur_blocks[j]);
        continue;
      }

      std::vector<int> clique;
      clique = solvePCM(ov_info_);
      double inlier_ratio = (double)clique.size() * 100.0 / ov_info_.size();
      printf("[MapMerger][SolvePCM] Max Clique / Info Size: %ld/%ld, Inlier Ratio: %lf% ... \n", clique.size(), ov_info_.size(), (double)clique.size() * 100.0 / ov_info_.size());


      if(inlier_ratio > 70.0) {
        std::sort(clique.begin(), clique.end());
        reduceVector(ov_info_, clique);
        stop = true;
        printf("[MapMerger] Complete the PR procedure!\n");
        break;
      }
      else {
        ref_set.erase(ref_blocks[i]);
        cur_set.erase(cur_blocks[j]);
        ov_info_.clear();
      }
    }
  }

  printf("Finally, found ov info size: %ld\n", ov_info_.size());
  if(ov_info_.size() < 3) {
    printf("[MultiMapMerger] Failed to merge submap, too few loop frames!\n");
    ov_info_.clear();
    map_cache_.push_back(cur_map_);
    return false;
  }

  std::stringstream ss;
  ss << setw(5) << setfill('0') << merge_num_;
  std::string num_str;
  ss >> num_str;

  for(auto info: ov_info_) {
    viewer_->AddLoopEdge(std::make_pair(info.bi->getHostFrame(), info.bj->getHostFrame()));
  }  
  // cv::imshow("bottom", bottom);
  // cv::waitKey(0);

  Trc = solveAveragePose(ov_info_);
  cur_map_->transform(Trc);


  // cv::imshow("bottom", bottom);
  // cv::waitKey(0);
  viewer_->SetRefMap(nullptr);
  viewer_->SetCurMap(nullptr);


  std::string map_cur_name = output_path_ + "/submap/submap_" + num_str + ".json";
  std::string traj_cur_name = output_path_ + "/submap/submap_traj_" + num_str + ".txt";
  cur_map_->saveJsonFile(map_cur_name);
  // cur_map_->saveTumTrajectory(traj_cur_name);
  // viewer_->SetRefMap(nullptr);
  // viewer_->SetCurMap(nullptr);

  optimizePGO(ref_map_, cur_map_, ov_info_);
  // viewer_->SetRefMap(ref_map_);
  // cv::imshow("bottom", bottom);
  // cv::waitKey(0);

  // viewer_->SetRefMap(nullptr);
  combineMap();
  merge_num_++;

  cur_map_->releaseAll();

  viewer_->SetRefMap(ref_map_);
  viewer_->SetCurMap(nullptr);
  // cv::imshow("bottom", bottom);
  // cv::waitKey(0);
  viewer_->SetRefMap(nullptr);
  viewer_->RemoveLoopEdge();


  std::string map_pgo_name = output_path_ + "/pgo/merge_map_pgo_" + num_str + ".json";
  std::string traj_pgo_name = output_path_ + "/pgo/merge_traj_pgo_" + num_str + ".txt";
  ref_map_->saveJsonFile(map_pgo_name);
  // ref_map_->saveTumTrajectory(traj_pgo_name);

  ref_map_->ba_time_ = 0;
  ref_map_->nfr_time_ = 0;
  ref_map_->optimizeFullBA();
  ref_map_->prune(1.414);
  ref_map_->optimizeFullBA();

  viewer_->SetRefMap(ref_map_);
  printf("Visualization 3\n");
  cv::imshow("bottom", bottom);
  cv::waitKey(0);
  viewer_->SetRefMap(nullptr);

  ref_map_->extractSparseStruct();

  std::string map_ba_name = output_path_ + "/ba/merge_map_ba_" + num_str + ".json";
  std::string traj_ba_name = output_path_ + "/ba/merge_traj_ba_" + num_str + ".txt";
  std::string summary_name = output_path_ + "/summary/merge_" + num_str + ".json";
  ref_map_->saveJsonFile(map_ba_name);
  // ref_map_->saveTumTrajectory(traj_ba_name);

  ref_map_->saveSummary(summary_name);

  viewer_->SetRefMap(ref_map_);
  // cv::imshow("bottom", bottom);
  // cv::waitKey(0);
  viewer_->SetRefMap(nullptr);
  return true;
}

void MultiMapMerger::findLoopMS() {
  
  loop_info_.clear();
  ov_info_.clear();

  block_vec_.clear();
  graph_vec_.clear();
  block_vec_.resize(map_cache_.size());
  graph_vec_.resize(map_cache_.size());
  // for(auto &map: map_cache_) {

  for(int mid = 0; mid < map_cache_.size(); mid++) {
    auto& map = map_cache_[mid];
    std::cout << mid << std::endl;
    map->divideBlocks(20.0f, 50.0f);
    std::vector<Block::Ptr> blocks = map->getBlocks();
    std::cout << "map id: " << mid << " block size: " << blocks.size() << std::endl;

    for(auto block: blocks) {
      Frame::Ptr frame = block->getHostFrame();
      block_vec_[mid].push_back(block);

      SceneGraph::Ptr graph(new SceneGraph);
      graph->buildFromBlock(block);
      graph->buildSelfAffinity();
      graph_vec_[mid].push_back(graph);
      std::cout << "graph node size: " << graph->nodes_.size() << std::endl;
    }
    std::cout << "id: " << mid <<  " Done" << std::endl;
    std::cout << map_cache_.size() << std::endl;
  }


  cv::Mat bottom(100, 100, CV_8U, cv::Scalar(255));
  std::vector<std::pair<uint32_t, uint32_t>> vec_map_ids;
  std::vector<std::pair<VectorMap::Ptr, VectorMap::Ptr>> vec_map_pairs;
  std::vector<std::vector<OverlapInfo>> vec_infos;

  for(int index_i = 0; index_i < map_cache_.size(); ++index_i) {
    for(int index_j = index_i + 1; index_j < map_cache_.size(); ++index_j) {
      printf("[MultiMapMerger] findLoopMS between (%d, %d)\n", index_i, index_j);
      std::vector<OverlapInfo> ov_infos;
      bool status = findPairLoop(map_cache_[index_i], map_cache_[index_j], \
                                 block_vec_[index_i], block_vec_[index_j], \
                                 graph_vec_[index_i], graph_vec_[index_j], ov_infos);

      if(status) {
        vec_map_pairs.push_back(std::make_pair(map_cache_[index_i], map_cache_[index_j]));
        vec_infos.push_back(ov_infos);
        vec_map_ids.push_back(std::make_pair(index_i, index_j));
        // ov_info_map_[map_cache_[index_i]][map_cache_[index_j]] = ov_infos;
      }
    }
  }

  if(vec_map_pairs.size() == 0) {
    printf("There is no loop frames between these map!\n");
    return;
  }

  Graph max_graph;
  for(auto id_info: vec_map_ids) {
    max_graph[id_info.first].insert(id_info.second);
    max_graph[id_info.second].insert(id_info.first);
  }


  // std::vector<int> connected_set = findMaxConnectedGraph(map_nodes);
  std::vector<int> connected_set = findConnectedComponents(max_graph);
  // std::sort(connected_set.begin(), connected_set.end());
  std::cout << "map cache: " << map_cache_.size() << std::endl;
  std::cout << "map clique: " << connected_set.size() << std::endl;

  map_queue_.clear();
  for(auto index: connected_set) {
    map_queue_.push_back(map_cache_[index]);
    viewer_->InsertMap(map_cache_[index]);
  }

  
  ov_info_map_.clear(); 
  ov_info_.clear();
  for(int i = 0; i < vec_map_ids.size(); i++) {
    auto mi = vec_map_pairs[i].first, mj = vec_map_pairs[i].second;
    
    std::vector<VectorMap::Ptr>::iterator iter0, iter1;
    iter0 = std::find(map_queue_.begin(), map_queue_.end(), mi);
    iter1 = std::find(map_queue_.begin(), map_queue_.end(), mj);
    if(iter0 != map_queue_.end() && iter1 != map_queue_.end()) {
      ov_info_.insert(ov_info_.end(), vec_infos[i].begin(), vec_infos[i].end());
      ov_info_map_[mi][mj] = vec_infos[i];
      std::cout << "map i: " << vec_map_ids[i].first << " map j: " << vec_map_ids[i].second << " size: " << vec_infos[i].size() << std::endl;
    }
  }
  for(auto info: ov_info_) {
    viewer_->AddLoopEdge(std::make_pair(info.bi->getHostFrame(), info.bj->getHostFrame()));
  }  


  std::set<VectorMap::Ptr> rmap_set, urmap_set;
  std::map<VectorMap::Ptr, Transform> rmap_tf;
  for(auto map: map_queue_) {
    urmap_set.insert(map);
  }
  urmap_set.erase(map_queue_.front());
  rmap_set.insert(map_queue_.front());
  rmap_tf[map_queue_.front()] = Transform();

  while(urmap_set.size() > 0) {
    bool exit = true;

    for(auto iter0: ov_info_map_) {
      auto mi = iter0.first;
      for(auto iter1: iter0.second) {
        auto mj = iter1.first;
        bool ri = (rmap_set.find(mi) != rmap_set.end());
        bool rj = (rmap_set.find(mj) != rmap_set.end());

        if(!ri && !rj) {
        continue;
        }
        else if(ri && !rj) {
          Transform Trc = solveAveragePose(iter1.second);
          Transform Twm = rmap_tf[mi];
          // cv::imshow("bottom", bottom);
          // cv::waitKey(0);   

          std::cout << "Pose: " << std::endl << (Twm * Trc).matrix() << std::endl;        
          mj->transform(Twm * Trc);
          rmap_set.insert(mj);
          urmap_set.erase(mj);
          rmap_tf[mj] = Twm * Trc;
          exit = false;

          // cv::imshow("bottom", bottom);
          // cv::waitKey(0);
        }
        else if(!ri && rj) {
          Transform Trc = solveAveragePose(iter1.second).inverse();
          Transform Twm = rmap_tf[mj];
          std::cout << "Pose: " << std::endl << (Twm * Trc).matrix() << std::endl;      
          cv::imshow("bottom", bottom);
          cv::waitKey(0);   

          mi->transform(Twm * Trc);
          rmap_set.insert(mi);
          urmap_set.erase(mi);
          rmap_tf[mi] = Twm * Trc;
          exit = false;

          // cv::imshow("bottom", bottom);
          // cv::waitKey(0);
        }
        else if(ri && rj) {
          continue;
        }
        printf("unregistered set: %ld, registered set: %ld, full size: %ld\n", urmap_set.size(), rmap_set.size(), map_queue_.size());
      }
    }
    if(exit)
      break;
  }

  optimizePGO(ov_info_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);
  for(auto &map: map_queue_) {
    combineMap(map);
    // combineMap(base_map_, map);
  }
  viewer_->ClearMapBuffer();
  viewer_->SetRefMap(base_map_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);

  base_map_->saveTrajAndCloudPath("../merge_wf_pgo.txt");
  base_map_->saveTumTrajectory("../merge_pgo.txt");

  base_map_->checkSelf();
  base_map_->optimizeFullBA();
  base_map_->prune(1.414);
  base_map_->optimizeFullBA();
  base_map_->saveTrajAndCloudPath("../merge_wf_ba.txt");
  base_map_->saveTumTrajectory("../merge_ba.txt");
  base_map_->saveJsonFile("../merge_map.json");
  viewer_->SetRefMap(base_map_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);

  // viewer_->SetRefMap(nullptr);
  // base_map_->extractSparseStruct();
  // viewer_->SetRefMap(base_map_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);

}




Block::Ptr MultiMapMerger::initBlock(const Frame::Ptr frame, const VectorMap::Ptr map, const float range) {
  Block::Ptr block(new Block);
  block->setHostFrame(frame);

  auto line_lms = map->getSemLines();
  auto surf_lms = map->getSemSurfaces();
  auto line_struct = map->getSemLineStruct();
  auto surf_struct = map->getSemSurfStruct();

  Eigen::Vector3d const pos = frame->Twb().p();
  pcl::PointXYZL index;
  index.getVector3fMap() = pos.cast<float>();
  uint64_t line_size = 0, surf_size = 0;
  for(auto &cls_iter: line_struct) {
    std::vector<int> indices;
    std::vector<float> distances;
    if(cls_iter.second.first->empty()) continue;
    cls_iter.second.second->radiusSearch(index, range, indices, distances);
    for(int i = 0; i < indices.size(); ++i) {
      uint32_t lm_id = cls_iter.second.first->points[indices[i]].label;
      block->insertLine(line_lms[cls_iter.first][lm_id]);
      line_size++;
    }
  }
  for(auto &cls_iter: surf_struct) {
    std::vector<int> indices;
    std::vector<float> distances;
    if(cls_iter.second.first->empty()) continue;
    cls_iter.second.second->radiusSearch(index, range, indices, distances);
    for(int i = 0; i < indices.size(); ++i) {
      uint32_t lm_id = cls_iter.second.first->points[indices[i]].label;
      block->insertSurface(surf_lms[cls_iter.first][lm_id]);
      surf_size++;
    }
  }
  block->initStruct();
  printf("[MultiMapMerger] initBlock from frame id: %d, Line: %ld, Surf: %ld\n", frame->id(), line_size, surf_size);
  return block;
}


bool MultiMapMerger::solveRelativePose(const SceneGraph::Ptr& gref, const SceneGraph::Ptr& gcur, 
                                  const clipper::Association& matches, Transform& Trc) {
  // coarse registraton (using matches)
  TicToc timer;
  const int size = matches.rows();
  ceres::Problem problem;
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  
  Trc.SetIdentity();

  problem.AddParameterBlock(Trc.parameters().data(), 7, pose_local_param);

  for(int i = 0; i < size; ++i) {
    SceneGraphNode::Ptr const& ref_node = gref->nodes_[matches(i, 0)];
    SceneGraphNode::Ptr const& cur_node = gcur->nodes_[matches(i, 1)];
    auto const geo_type = ref_node->geometry_type();
    if(geo_type == SceneGraphNode::GeoType::LINE) {
      LaserPointToLineFactor *f = new LaserPointToLineFactor(cur_node->node_value(), ref_node->node_value(), ref_node->normal_value(), 1/0.2);
      problem.AddResidualBlock(f, huber_loss, Trc.parameters().data());
    }
    else if(geo_type == SceneGraphNode::GeoType::SURFACE) {
      LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(cur_node->node_value(), ref_node->node_value(), ref_node->normal_value(), 1/0.2);
      problem.AddResidualBlock(f, huber_loss, Trc.parameters().data());
    }
  }
  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 50;
  options.function_tolerance = 1e-4;
  options.gradient_tolerance = 1e-4;
  // options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Eigen::Vector3d const euler = R2ypr(Trc.dcm());
  // std::cout << "euler: " << euler.transpose() << std::endl;
  double pitch = euler(1) * 180 / M_PI, roll = euler(2) * 180 / M_PI;
  if(std::abs(pitch) > 20.0 || std::abs(roll) > 20.0) {
    printf("[MapMerging][SolveRelativePose] Failed! Too large roll or pitch. (Pitch: %lf, Roll: %lf) Time Cost: %lf ms.\n", pitch, roll, timer.toc());
    return false;
  }
  
  if(summary.termination_type == ceres::TerminationType::CONVERGENCE) {
    // std::cout << summary.BriefReport() << std::endl;
    printf("[MapMerging][SolveRelativePose] Successful! Time Cost: %lf ms.\n", timer.toc());
    return true;
  }
  else {
    // std::cout << summary.BriefReport() << std::endl;
    printf("[MapMerging][SolveRelativePose] Failed! Time Cost: %lf ms.\n", timer.toc());
    return false;
  }
    
}

bool MultiMapMerger::refineRelativePose(const Block::Ptr& block_i, const Block::Ptr& block_j, Transform& Tij, Eigen::Matrix<double, 6, 6>& sqrt_info) {
  // refine pose (using knn search and semi-dense centroid cloud)
  // for every landmark in block j, find the nearest landmark (using the coarse Tij) in current map
  TicToc timer;
  auto &sem_line_lms_i = block_i->getSemLines();
  auto &sem_line_lms_j = block_j->getSemLines();
  auto &sem_surf_lms_i = block_i->getSemSurfaces();
  auto &sem_surf_lms_j = block_j->getSemSurfaces();
  auto &sem_line_struct_i = block_i->getSemLineStruct();
  auto &sem_surf_struct_i = block_i->getSemSurfStruct();

  double line_thres = 20.0, surface_thres = 20.0;
  int iter = 0;
  const int max_iter = 5;

  bool converge = false;
  double line_inlier_ratio = 0.0, surf_inlier_ratio = 0.0;
  Eigen::MatrixXd covariance(7, 7);
  for(; iter < max_iter; ++iter) {
    ceres::Problem problem;
    ceres::LossFunction *huber_loss = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();   
    problem.AddParameterBlock(Tij.parameters().data(), 7, pose_local_param);

    uint32_t line_lm_cnt = 0, surf_lm_cnt = 0, line_lm_valid_cnt = 0, surf_lm_valid_cnt = 0;
    for(auto &cls_iter: sem_line_lms_j) {
      int lm_size = cls_iter.second.size();
      if(lm_size == 0) continue;
      CentroidCloudPtr &nodes = sem_line_struct_i[cls_iter.first].first;
      if(nodes.get() == nullptr) continue;
      if(nodes->empty()) continue;
      CentroidKdTreePtr &tree = sem_line_struct_i[cls_iter.first].second;
      std::unordered_map<uint32_t, Block::Node::Ptr> &lms = sem_line_lms_i[cls_iter.first];

      for(auto &lm_iter: cls_iter.second) {
        Block::Node::Ptr lm_j = lm_iter.second;
        Eigen::Vector3d const cj = lm_j->centroid;
        Eigen::Vector3d const nj = lm_j->normal;
        Eigen::Vector3d const qj = Tij * cj;
        line_lm_cnt++;

        // nn search
        pcl::PointXYZL index;
        index.getVector3fMap() = qj.cast<float>();
        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(index, 1, indices, distances);
        if(indices.empty()) 
          continue;

        uint32_t const lm_id = nodes->points[indices[0]].label;
        Block::Node::Ptr lm_i = lms[lm_id];
        Eigen::Vector3d const ci = lm_i->centroid;
        Eigen::Vector3d const ni = lm_i->normal;
        Eigen::Matrix3d const nmat = Eigen::Matrix3d::Identity() - ni * ni.transpose();
        Eigen::Vector3d const r = nmat * (qj - ci);
        if(r.norm() < line_thres) {
          LaserPointToLineFactor *f = new LaserPointToLineFactor(cj, ci, ni, 1/0.2);
          problem.AddResidualBlock(f, huber_loss, Tij.parameters().data());
          line_lm_valid_cnt++;
        }
      }
    }

    for(auto &cls_iter: sem_surf_lms_j) {
      int lm_size = cls_iter.second.size();
      if(lm_size == 0) continue;
      CentroidCloudPtr &nodes = sem_surf_struct_i[cls_iter.first].first;
      if(nodes.get() == nullptr) continue;
      if(nodes->empty()) continue;
      CentroidKdTreePtr &tree = sem_surf_struct_i[cls_iter.first].second;
      std::unordered_map<uint32_t, Block::Node::Ptr> &lms = sem_surf_lms_i[cls_iter.first];

      for(auto &lm_iter: cls_iter.second) {
        Block::Node::Ptr lm_j = lm_iter.second;
        Eigen::Vector3d const cj = lm_j->centroid;
        Eigen::Vector3d const nj = lm_j->normal;
        Eigen::Vector3d const qj = Tij * cj;
        surf_lm_cnt++;

        // nn search
        pcl::PointXYZL index;
        index.getVector3fMap() = qj.cast<float>();
        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(index, 1, indices, distances);
        if(indices.empty()) 
          continue;

        uint32_t const lm_id = nodes->points[indices[0]].label;
        Block::Node::Ptr lm_i = lms[lm_id];

        Eigen::Vector3d const ci = lm_i->centroid;
        Eigen::Vector3d const ni = lm_i->normal;
        SurfaceInfo surface(ci, ni);
        double const ndist = surface.distance(qj);
        Eigen::Vector3d const pedal = surface.pedal(qj);
        double const pdist = (ci - pedal).norm();
        double const theta = std::acos(ni.dot(Tij.dcm() * nj));
        // if(pdist < GetSemanticResolution(cls_iter.first) && std::abs(ndist) < surface_thres) {
        if(pdist < lm_i->radius * 2 && std::abs(ndist) < surface_thres) {
          LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(cj, ci, ni, 1/0.2);
          problem.AddResidualBlock(f, huber_loss, Tij.parameters().data());
          surf_lm_valid_cnt++;
        }
      }
    }
    line_thres = std::max(0.3, line_thres/4.0);
    surface_thres = std::max(0.3, surface_thres/4.0);

    if(problem.NumResidualBlocks() < 30) {
      printf("[MapMerging][RefineRelativePose] Failed! Too little correspondences! (%d)...\n", problem.NumResidualBlocks());
      return false;
    }


    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.BriefReport() << std::endl;
    line_inlier_ratio = (line_lm_cnt == 0) ? 1 : (double)line_lm_valid_cnt/line_lm_cnt;
    surf_inlier_ratio = (surf_lm_cnt == 0) ? 0 : (double)surf_lm_valid_cnt/surf_lm_cnt;

    Eigen::Vector3d const euler = R2ypr(Tij.dcm());
    if(std::abs(euler(1)) > 20.0 / 180.0 * M_PI || std::abs(euler(2)) > 20.0 / 180.0 * M_PI) {
      // printf("[MapMerging][RefineRelativePose] Failed! Too large roll or pitch. (Pitch: %lf, Roll: %lf) Time Cost: %lf ms.\n", pitch, roll, timer.toc());
      return false;
    }
  }
  if(line_inlier_ratio < 0.4 || surf_inlier_ratio < 0.4) {
    printf("[MapMerging][RefineRelativePose] Failed! Too little landmark registration inlier ratio! (%lf, %lf)...\n", line_inlier_ratio, surf_inlier_ratio);
    return false;
  }
  // std::cout << "[MapMerging][RefineRelativePose] Successful! Refine Pose: " << Tij.p().transpose() << " " << R2ypr(Tij.dcm()).transpose() << std::endl;
  printf("[MapMerging][RefineRelativePose] Successful! Line/Surface Ratio: %lf/%lf%\n", line_inlier_ratio * 100, surf_inlier_ratio * 100);
  // printf("[MapMerging][RefineRelativePose] Successful! Time Cost: %lf ms. Iteration: %d.\n", timer.toc(), iter);
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(covariance.block<6, 6>(0, 0));
  Eigen::VectorXd Sinvsq = Eigen::VectorXd(saes.eigenvalues().array().inverse()).cwiseSqrt();
  sqrt_info =  saes.eigenvectors() * Sinvsq.asDiagonal() * saes.eigenvectors().transpose();
  sqrt_info = Eigen::Matrix<double, 6, 6>::Identity();
  sqrt_info.block<3, 3>(0, 0) *= 1/0.005;
  sqrt_info.block<3, 3>(3, 3) *= 1/0.001;
  return true;
}

bool MultiMapMerger::findPairLoop(const VectorMap::Ptr& mi, const VectorMap::Ptr& mj,
      const std::vector<Block::Ptr>& blocks0, const std::vector<Block::Ptr>& blocks1,
      const std::vector<SceneGraph::Ptr>& graphs0, const std::vector<SceneGraph::Ptr>& graphs1, std::vector<OverlapInfo>& ov_infos) {

  std::set<Block::Ptr> ref_set, cur_set;
  std::vector<std::pair<Frame::Ptr, Frame::Ptr>> outlier_loop_info;
  bool stop = false;
  ov_infos.clear();
  Transform Trc;
  for(int i = 0; i < graphs0.size(); ++i) {
    if(stop) break;
    if(ref_set.find(blocks0[i]) != ref_set.end())
      continue;
    for(int j = 0; j < graphs1.size(); ++j) {
      if(cur_set.find(blocks1[j]) != cur_set.end())
        continue;

      GlobalRegister global_reg;
      printf("[MultiMapMerger] Call Global Registration. Node Size: (%ld, %ld)\n", graphs0[i]->nodes_.size(), graphs1[j]->nodes_.size());
      auto association = global_reg.solveMatch(graphs0[i], graphs1[j]);

      double ref_ratio = (double)association.rows() / graphs0[i]->nodes_.size();
      double cur_ratio = (double)association.rows() / graphs1[j]->nodes_.size();
      if(ref_ratio < 0.5 || cur_ratio < 0.5) {
        printf("[MapMerger] Too low overlap ratio! (%lf, %lf)\n", ref_ratio, cur_ratio);
        continue;
      }

      Transform Tij;
      Eigen::Matrix<double, 6, 6> sqrt_info;
      bool suc = false;
      if(!solveRelativePose(graphs0[i], graphs1[j], association, Tij)) {
        printf("[MapMerger] Failed in blocks coarse align!\n");
        // outlier_loop_info.push_back(std::make_pair(bi->getHostFrame(), bj->getHostFrame()));
        continue;
      }

      if(!refineRelativePose(blocks0[i], blocks1[j], Tij, sqrt_info)) {
      // if(!refineRelativePoseGNC(ref_blocks[i], cur_blocks[j], Tij)) {
        printf("[MapMerger] Failed in blocks refine align!\n");
        // outlier_loop_info.push_back(std::make_pair(ref_blocks[i]->getHostFrame(), cur_blocks[j]->getHostFrame()));
        continue;
      }

      const Transform Twbi = blocks0[i]->getHostFrame()->Twb();
      const Transform Twbj = blocks1[j]->getHostFrame()->Twb();
      auto first_info = OverlapInfo(blocks0[i], blocks1[j], mi, mj, Twbi.inverse() * Tij * Twbj, Tij, sqrt_info);
      ov_infos.push_back(first_info);
      ref_set.insert(blocks0[i]);
      cur_set.insert(blocks1[j]);

      Trc = Tij;
      printf("[MapMerger][Merge] System has found the first overlap and starts to find more potential overlap ...\n");
      std::vector<OverlapInfo> pot_ov_infos = searchPotentialOverlap(blocks0, blocks1, ref_set, cur_set, Trc);
      ov_infos.insert(ov_infos.end(), pot_ov_infos.begin(), pot_ov_infos.end());

      if(ov_infos.size() < 5) {
        printf("[MapMerger] There are %ld loop frames, reject!\n", ov_infos.size());
        ov_infos.clear();
        ref_set.erase(blocks0[i]);
        cur_set.erase(blocks1[j]);
        continue;
      }

      std::vector<int> clique;
      clique = solvePCM(ov_infos);
      double inlier_ratio = (double)clique.size() * 100.0 / ov_infos.size();
      printf("[MapMerger][SolvePCM] Max Clique / Info Size: %ld/%ld, Inlier Ratio: %lf% ... \n", clique.size(), ov_infos.size(), (double)clique.size() * 100.0 / ov_infos.size());


      if(inlier_ratio > 70.0) {
        std::sort(clique.begin(), clique.end());
        reduceVector(ov_infos, clique);
        stop = true;
        printf("[MapMerger] Complete the PR procedure!\n");
        break;
      }
      else {
        ref_set.erase(blocks0[i]);
        cur_set.erase(blocks1[j]);
        ov_infos.clear();
      }
    }
  }
  printf("Finally, found ov info size: %ld\n", ov_infos.size());
  if(ov_infos.size() < 3) {
    return false;
  }
  return true;
}

void MultiMapMerger::selectLoop(std::vector<LoopInfo>& loop_info, std::vector<OverlapInfo>& ov_info) {
  ov_info.clear();
  printf("[MultiMapMerger] selectLoop have %ld loop info ... \n", loop_info.size());
  for(auto info: loop_info) {
    VectorMap::Ptr mi = info.mi, mj = info.mj;
    Frame::Ptr fi = info.rel_pose_info.fi, fj = info.rel_pose_info.fj;
    Block::Ptr bi = block_map_[fi], bj = block_map_[fj];
    SceneGraph::Ptr gi = graph_map_[fi], gj = graph_map_[fj];
    printf("[MultiMapMerger] Process loop pair (%d, %d) with nodes (%ld, %ld)\n", fi->id(), fj->id(), gi->nodes_.size(), gj->nodes_.size());

    GlobalRegister global_reg;
    auto association = global_reg.solveMatch(gi, gj);

    double ref_ratio = (double)association.rows() / gi->nodes_.size();
    double cur_ratio = (double)association.rows() / gj->nodes_.size();
    if(ref_ratio < 0.5 || cur_ratio < 0.5) {
      // printf("[MapMerger] Too low overlap ratio! (%lf, %lf)\n", ref_ratio, cur_ratio);
      continue;
    }

    Transform Tij;
    Eigen::Matrix<double, 6, 6> sqrt_info;
    bool suc = false;
    if(!solveRelativePose(gi, gj, association, Tij)) {
      // printf("[MapMerger] Failed in blocks coarse align!\n");
      // outlier_loop_info.push_back(std::make_pair(bi->getHostFrame(), bj->getHostFrame()));
      continue;
    }

    if(!refineRelativePose(bi, bj, Tij, sqrt_info)) {
    // if(!refineRelativePoseGNC(ref_blocks[i], cur_blocks[j], Tij)) {
      // printf("[MapMerger] Failed in blocks refine align!\n");
      // outlier_loop_info.push_back(std::make_pair(ref_blocks[i]->getHostFrame(), cur_blocks[j]->getHostFrame()));
      continue;
    }
    // break; 
    const Transform Twbi = bi->getHostFrame()->Twb();
    const Transform Twbj = bj->getHostFrame()->Twb();
    OverlapInfo oinfo(bi, bj, mi, mj, Twbi.inverse() * Tij * Twbj, Tij, sqrt_info);
    ov_info.push_back(oinfo);
  }
  printf("[MapMerger][SolvePCM] Block Size: %ld ...\n", ov_info.size());
  std::vector<int> clique = solvePCM(ov_info);
  double inlier_ratio = (double)clique.size() * 100.0 / ov_info.size();
  printf("[MapMerger][SolvePCM] Max Clique / Info Size: %ld/%ld, Inlier Ratio: %lf% \n", clique.size(), ov_info.size(), (double)clique.size() * 100.0 / ov_info.size());
  reduceVector(ov_info, clique);
}

void MultiMapMerger::alignMultiMap() {

  cv::Mat bottom(100, 100, CV_8U, cv::Scalar(255));
  for(auto& info: loop_info_) {
    loop_info_map_[info.mi][info.mj].push_back(info);
    std::cout << "map i: " << (uint64_t)info.mi.get() << " map j: " << (uint64_t)info.mj.get() << " size: " << loop_info_map_[info.mi][info.mj].size() << std::endl;
  }

  std::vector<MapNode::Ptr> map_nodes;
  for(int i = 0; i < map_cache_.size(); ++i) {
    MapNode::Ptr node(new MapNode(i, map_cache_[i]));
    map_nodes.push_back(node);
  }
  
  for(auto linfo: loop_info_) {
    std::vector<VectorMap::Ptr>::iterator mi_iter = std::find(map_cache_.begin(), map_cache_.end(), linfo.mi); 
    auto mi_index = std::distance(map_cache_.begin(), mi_iter);
    std::vector<VectorMap::Ptr>::iterator mj_iter = std::find(map_cache_.begin(), map_cache_.end(), linfo.mj); 
    auto mj_index = std::distance(map_cache_.begin(), mj_iter);
    map_nodes[mi_index]->neighbors.push_back(map_nodes[mj_index]);
    map_nodes[mj_index]->neighbors.push_back(map_nodes[mi_index]);
  }

  std::vector<int> connected_set = findMaxConnectedGraph(map_nodes);
  std::sort(connected_set.begin(), connected_set.end());
  std::cout << "map cache: " << map_cache_.size() << std::endl;
  std::cout << "map clique: " << connected_set.size() << std::endl;

  map_queue_.clear();
  for(auto index: connected_set) {
    map_queue_.push_back(map_cache_[index]);
    viewer_->InsertMap(map_cache_[index]);
  }
  
  cv::imshow("bottom", bottom);
  cv::waitKey(0);
  
  ov_info_.clear();
  printf("Map Queue Size: %ld ...\n", map_queue_.size());
  std::map<uint64_t, std::pair<VectorMap::Ptr, VectorMap::Ptr>> loop_num_map;

  std::vector<std::pair<VectorMap::Ptr, VectorMap::Ptr>> vec_frame_pairs;
  std::vector<std::vector<OverlapInfo>> vec_infos;

  for(int index_i = 0; index_i < map_queue_.size(); index_i++) {
    for(int index_j = index_i + 1; index_j < map_queue_.size(); index_j++) {
      std::vector<OverlapInfo> ov_info;
      auto& loop_info = loop_info_map_[map_queue_[index_i]][map_queue_[index_j]];
      printf("(%d, %d) Size: %ld\n", index_i, index_j, loop_info.size());
      std::cout << "map 0: " << (uint64_t)map_queue_[index_i].get() << " map 1: " << map_queue_[index_j].get() << std::endl;
      selectLoop(loop_info, ov_info);
      if(ov_info.size() > 0) {
        // ov_info_map_[map_queue_[index_i]][map_queue_[index_j]] = ov_info;
        // loop_num_map[ov_info.size()] = std::make_pair(map_queue_[index_i], map_queue_[index_j]);  
        ov_info_.insert(ov_info_.end(), ov_info.begin(), ov_info.end());    
        vec_frame_pairs.push_back(std::make_pair(map_queue_[index_i], map_queue_[index_j]));
        vec_infos.push_back(ov_info);
      }
    }
  }
  
  for(auto info: ov_info_) {
    viewer_->AddLoopEdge(std::make_pair(info.bi->getHostFrame(), info.bj->getHostFrame()));
  }  
  cv::imshow("bottom", bottom);
  cv::waitKey(0);


  std::set<VectorMap::Ptr> rmap_set, urmap_set;
  std::map<VectorMap::Ptr, Transform> rmap_tf;
  for(auto map: map_queue_) {
    urmap_set.insert(map);
  }
  urmap_set.erase(map_queue_.front());
  rmap_set.insert(map_queue_.front());
  rmap_tf[map_queue_.front()] = Transform();

  while(urmap_set.size() > 0) {
    bool exit = true;
    for(int index = 0; index < vec_frame_pairs.size(); ++index) {
      VectorMap::Ptr mi = vec_frame_pairs[index].first, mj = vec_frame_pairs[index].second;

      std::vector<VectorMap::Ptr>::iterator mi_iter = std::find(map_cache_.begin(), map_cache_.end(), mi); 
      auto mi_index = std::distance(map_cache_.begin(), mi_iter);
      std::vector<VectorMap::Ptr>::iterator mj_iter = std::find(map_cache_.begin(), map_cache_.end(), mj); 
      auto mj_index = std::distance(map_cache_.begin(), mj_iter);
      printf("Mi: %ld, Mj: %ld\n", mi_index, mj_index);


      bool ri = (rmap_set.find(mi) != rmap_set.end());
      bool rj = (rmap_set.find(mj) != rmap_set.end());
      if(!ri && !rj) {
        continue;
      }
      else if(ri && !rj) {
        Transform Trc = solveAveragePose(vec_infos[index]);
        Transform Twm = rmap_tf[mi];
        mj->transform(Twm * Trc);
        rmap_set.insert(mj);
        urmap_set.erase(mj);
        rmap_tf[mj] = Twm * Trc;
        exit = false;
        std::cout << "transform id: " << mj_index << std::endl;
        std::cout << "Pose: " << std::endl << (Twm * Trc).matrix() << std::endl;
        cv::imshow("bottom", bottom);
        cv::waitKey(0);
      }
      else if(!ri && rj) {
        Transform Trc = solveAveragePose(vec_infos[index]).inverse();
        Transform Twm = rmap_tf[mj];
        mi->transform(Twm * Trc);
        rmap_set.insert(mi);
        urmap_set.erase(mi);
        rmap_tf[mi] = Twm * Trc;
        exit = false;
        std::cout << "transform id: " << mi_index << std::endl;
        std::cout << "Pose: " << std::endl << (Twm * Trc).matrix() << std::endl;
        cv::imshow("bottom", bottom);
        cv::waitKey(0);
      }
      else if(ri && rj) {
        continue;
      }
      printf("unregistered set: %ld, registered set: %ld, full size: %ld\n", urmap_set.size(), rmap_set.size(), map_queue_.size());
    }
    if(exit)
      break;
  }

  cv::imshow("bottom", bottom);
  cv::waitKey(0);

  viewer_->ClearMapBuffer();
  map_queue_.clear();

  for(auto iter: rmap_set) {
    map_queue_.push_back(iter);
    viewer_->AddVectorMap(iter);
  }

  optimizePGO(ov_info_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);
  viewer_->ClearMapBuffer();

  for(auto map: map_queue_) {
    combineMap(map);
  }
  viewer_->SetRefMap(base_map_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);
  viewer_->SetRefMap(nullptr);
  base_map_->saveJsonFile("/home/summervibe/Desktop/uav_ws/src/uav_lsm/map_creator/map/helipr/base_map.json");

  optimizeGBA();
  base_map_->saveTumTrajectory("/home/summervibe/Desktop/uav_ws/src/uav_lsm/map_creator/exp/helipr/pgo.txt");
  base_map_->optimizeFullBA(1.0);
  base_map_->saveTumTrajectory("/home/summervibe/Desktop/uav_ws/src/uav_lsm/map_creator/exp/helipr/ba.txt");
  base_map_->extractSparseStruct();

  viewer_->SetRefMap(base_map_);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);


}


std::vector<int> MultiMapMerger::solvePCM(const std::vector<OverlapInfo>& info) {
  AssociationGraph graph;
  graph.build(info);
  MaxCliqueSolver solver;
  auto clique = solver.findMaxClique(graph);
  std::sort(clique.begin(), clique.end());
  return clique;
}

std::vector<OverlapInfo> MultiMapMerger::searchPotentialOverlap(const std::vector<Block::Ptr>& ref_blocks, 
                                                          const std::vector<Block::Ptr>& cur_blocks,
                                                          const std::set<Block::Ptr>& ref_set,
                                                          const std::set<Block::Ptr>& cur_set,
                                                          const Transform& Trc) {
  std::vector<OverlapInfo> info;
  std::set<Block::Ptr> _ref_set = ref_set, _cur_set = cur_set;

  for(const Block::Ptr& ref: ref_blocks) {
    double min_dist = 1e10;
    Block::Ptr ncur;
    if(_ref_set.find(ref) != _ref_set.end())
      continue;
    for(const Block::Ptr& cur: cur_blocks) {
      if(_cur_set.find(cur) != _cur_set.end())
        continue;
      Eigen::Vector3d const cref = ref->getHostFrame()->Twb().p();
      Eigen::Vector3d const ccur = Trc * cur->getHostFrame()->Twb().p();
      double dist = (cref - ccur).norm();
      if(dist < 20.0 && dist < min_dist) {
        ncur = cur;
        min_dist = dist;
      }
    }
    if(ncur == nullptr)
      continue;
    Transform Trel = Trc;
    Eigen::Matrix<double, 6, 6> sqrt_info;
    bool suc = refineRelativePose(ref, ncur, Trel, sqrt_info);
    if(suc) {
      info.push_back(OverlapInfo(ref, ncur, ref->getHostFrame()->Twb().inverse() * Trel * ncur->getHostFrame()->Twb(), Trel, sqrt_info));
      _ref_set.insert(ref);
      _cur_set.insert(ncur);
      continue;
    }
  }
  return info;
}



void MultiMapMerger::optimizePGO(const VectorMap::Ptr& ref_map, const VectorMap::Ptr& cur_map, const std::vector<OverlapInfo>& overlap) {

  // pose graph optimization (for coarse alignment)
  printf("[MultiMapMerger][optimizePGO] Start Pose Graph Optimization! ...\n");
  ceres::Problem problem;
  // ceres::LossFunction *huber_loss = new ceres::ArctanLoss(1.0);
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  

  auto &ref_kfs = ref_map->getKeyFrames();
  for(auto &kf_iter: ref_kfs) {
    Transform& Twb = kf_iter.second->Twb();
    problem.AddParameterBlock(Twb.parameters().data(), 7, pose_local_param);
  }
  auto &cur_kfs = ref_map->getKeyFrames();
  for(auto &kf_iter: cur_kfs) {
    Transform& Twb = kf_iter.second->Twb();
    problem.AddParameterBlock(Twb.parameters().data(), 7, pose_local_param);
  }

  Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity() * 1e8;
  PriorPoseFactor *prior_factor = new PriorPoseFactor(ref_map_->pivot_kf_->Twb(), sqrt_info);
  problem.AddResidualBlock(prior_factor, nullptr, ref_map_->pivot_kf_->Twb().parameters().data());
  
  std::vector<RelPoseInfo> ref_odom_infos = ref_map->getOdomInfo();
  for(auto &info: ref_odom_infos) {
    Frame::Ptr const fk = info.fi;
    Frame::Ptr const fl = info.fj;
    Transform const& Tkl = info.Tij;
    RelativePoseFactor *rel_pose_factor = new RelativePoseFactor(Tkl, info.sqrt_info);
    problem.AddResidualBlock(rel_pose_factor, huber_loss, 
      fk->Twb().parameters().data(), fl->Twb().parameters().data());
  }


  std::vector<RelPoseInfo> cur_odom_infos = cur_map->getOdomInfo();
  for(auto &info: cur_odom_infos) {
    Frame::Ptr const fk = info.fi;
    Frame::Ptr const fl = info.fj;
    Transform const& Tkl = info.Tij;
    RelativePoseFactor *rel_pose_factor = new RelativePoseFactor(Tkl, info.sqrt_info);
    problem.AddResidualBlock(rel_pose_factor, huber_loss, 
      fk->Twb().parameters().data(), fl->Twb().parameters().data());
  }
  
  for(auto &info: overlap) {
    Frame::Ptr const fk = info.bi->getHostFrame();
    Frame::Ptr const fl = info.bj->getHostFrame();
    Transform const& Tkl = info.Tij;
    RelativePoseFactor *rel_pose_factor = new RelativePoseFactor(Tkl, info.sqrt_info);
    problem.AddResidualBlock(rel_pose_factor, huber_loss, 
      fk->Twb().parameters().data(), fl->Twb().parameters().data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;
  options.function_tolerance = 1e-4;
  options.gradient_tolerance = 1e-4;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
}


void MultiMapMerger::optimizePGO(const std::vector<OverlapInfo>& overlap) {

  // pose graph optimization (for coarse alignment)
  printf("[MultiMapMerger][optimizePGO] Start Pose Graph Optimization! ...\n");
  ceres::Problem problem;
  // ceres::LossFunction *huber_loss = new ceres::ArctanLoss(1.0);
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  

  for(auto &map: map_queue_) {
    auto &kfs = map->getKeyFrames();
    for(auto &kf_iter: kfs) {
      Transform& Twb = kf_iter.second->Twb();
      problem.AddParameterBlock(Twb.parameters().data(), 7, pose_local_param);
      if(kf_iter.second->id() == 0 && map == map_queue_.front()) {
        problem.SetParameterBlockConstant(Twb.parameters().data());
      }
    }
  }

  for(auto &map: map_queue_) {
    std::vector<RelPoseInfo> odom_infos = map->getOdomInfo();
    for(auto &info: odom_infos) {
      Frame::Ptr const fk = info.fi;
      Frame::Ptr const fl = info.fj;
      Transform const& Tkl = info.Tij;
      RelativePoseFactor *rel_pose_factor = new RelativePoseFactor(Tkl, info.sqrt_info);
      problem.AddResidualBlock(rel_pose_factor, huber_loss, 
        fk->Twb().parameters().data(), fl->Twb().parameters().data());
    }
  }

  std::cout << "overlap size: " << overlap.size() << std::endl;
  for(auto &info: overlap) {
    Frame::Ptr const fk = info.bi->getHostFrame();
    Frame::Ptr const fl = info.bj->getHostFrame();
    Transform const& Tkl = info.Tij;
    RelativePoseFactor *rel_pose_factor = new RelativePoseFactor(Tkl, info.sqrt_info);
    problem.AddResidualBlock(rel_pose_factor, huber_loss, 
      fk->Twb().parameters().data(), fl->Twb().parameters().data());
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations = 50;
  options.function_tolerance = 1e-4;
  options.gradient_tolerance = 1e-4;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
}


void MultiMapMerger::combineMap(const VectorMap::Ptr& map) {

  if(base_map_ == nullptr) {
    base_map_ = VectorMap::Ptr(new VectorMap);
  }
  else {
    base_map_->resort();
  }

  auto &base_kfs = base_map_->getKeyFrames();
  auto &base_line_lms = base_map_->getSemLines();
  auto &base_surf_lms = base_map_->getSemSurfaces();
  auto &cur_kfs = map->getKeyFrames();
  auto &cur_line_lms = map->getSemLines();
  auto &cur_surf_lms = map->getSemSurfaces();

  if(base_map_->pivot_kf_ == nullptr) {
    base_map_->pivot_kf_ = map->pivot_kf_;
  }

  int frame_size = base_kfs.size();
  uint32_t new_id = 0;
  for(auto &f_iter: cur_kfs) {
    Frame::Ptr frame = f_iter.second;
    frame->setID(frame_size + new_id);
    assert(base_kfs.find(frame->id()) == base_kfs.end());
    base_map_->insertFrame(frame);
    new_id++; 
  }

  std::map<uint16_t, std::vector<LineLM::Ptr>> line_lms;
  std::map<uint16_t, std::vector<SurfaceLM::Ptr>> surf_lms;


  std::map<uint16_t, uint32_t> line_map_size;
  std::map<uint16_t, uint32_t> surf_map_size;
  std::map<uint16_t, uint32_t> line_map_nlm_size;
  std::map<uint16_t, uint32_t> surf_map_nlm_size;


  for(auto &cls_iter: base_line_lms) {
    line_map_size.insert(std::make_pair(cls_iter.first, cls_iter.second.size()));
    line_map_nlm_size.insert(std::make_pair(cls_iter.first, 0));
  }
  for(auto &cls_iter: base_surf_lms) {
    surf_map_size.insert(std::make_pair(cls_iter.first, cls_iter.second.size()));
    surf_map_nlm_size.insert(std::make_pair(cls_iter.first, 0));
  }

  for(auto &cls_iter: cur_line_lms) {
    uint16_t sem_type = cls_iter.first;
    int lm_size = cls_iter.second.size();
    if(lm_size == 0) continue;
    for(auto &cur_lm_iter: cls_iter.second) {
      LineLM::Ptr& cur_lm = cur_lm_iter.second;
      uint32_t new_id = line_map_size[cls_iter.first] + line_map_nlm_size[cls_iter.first];
      cur_lm->setID(new_id);     
      line_lms[cls_iter.first].push_back(cur_lm);
      line_map_nlm_size[cls_iter.first]++;
    }
  }

  for(auto &cls_iter: cur_surf_lms) {
    uint16_t sem_type = cls_iter.first;
    int lm_size = cls_iter.second.size();
    if(lm_size == 0) continue;
    for(auto &cur_lm_iter: cls_iter.second) {
      SurfaceLM::Ptr& cur_lm = cur_lm_iter.second;
      uint32_t new_id = surf_map_size[cls_iter.first] + surf_map_nlm_size[cls_iter.first];
      cur_lm->setID(new_id);     
      surf_lms[cls_iter.first].push_back(cur_lm);  
      surf_map_nlm_size[cls_iter.first]++;
    }
  }

  for(auto &cls_iter: line_lms) {
    for(auto &lm: cls_iter.second) {
      base_map_->insertLine(lm);
    }
  }
  for(auto &cls_iter: surf_lms) {
    for(auto &lm: cls_iter.second) {
      base_map_->insertSurface(lm);
    }
  }

  const auto& odom_info = map->getOdomInfo();
  for(const auto& info: odom_info) {
    base_map_->addOdomInfo(info);
  }
    
  base_map_->resort();
  base_map_->initStruct();
  base_map_->prune(1.414); 
}


void MultiMapMerger::combineMap() {

  ref_map_->resort();
  
  auto &ref_kfs = ref_map_->getKeyFrames();
  auto &ref_line_lms = ref_map_->getSemLines();
  auto &ref_surf_lms = ref_map_->getSemSurfaces();
  auto &cur_kfs = cur_map_->getKeyFrames();
  auto &cur_line_lms = cur_map_->getSemLines();
  auto &cur_surf_lms = cur_map_->getSemSurfaces();


  int frame_size = ref_kfs.size();
  uint32_t new_id = 0;
  for(auto &f_iter: cur_kfs) {
    Frame::Ptr frame = f_iter.second;
    frame->setID(frame_size + new_id);
    assert(ref_kfs.find(frame->id()) == ref_kfs.end());
    ref_map_->insertFrame(frame);
    new_id++; 
  }

  std::map<uint16_t, std::vector<LineLM::Ptr>> line_lms;
  std::map<uint16_t, std::vector<SurfaceLM::Ptr>> surf_lms;

  std::map<uint16_t, uint32_t> line_map_size;
  std::map<uint16_t, uint32_t> surf_map_size;
  std::map<uint16_t, uint32_t> line_map_nlm_size;
  std::map<uint16_t, uint32_t> surf_map_nlm_size;


  for(auto &cls_iter: ref_line_lms) {
    line_map_size.insert(std::make_pair(cls_iter.first, cls_iter.second.size()));
    line_map_nlm_size.insert(std::make_pair(cls_iter.first, 0));
  }
  for(auto &cls_iter: ref_surf_lms) {
    surf_map_size.insert(std::make_pair(cls_iter.first, cls_iter.second.size()));
    surf_map_nlm_size.insert(std::make_pair(cls_iter.first, 0));
  }

  for(auto &cls_iter: cur_line_lms) {
    uint16_t sem_type = cls_iter.first;
    int lm_size = cls_iter.second.size();
    if(lm_size == 0) continue;
    for(auto &cur_lm_iter: cls_iter.second) {
      LineLM::Ptr& cur_lm = cur_lm_iter.second;
      uint32_t new_id = line_map_size[cls_iter.first] + line_map_nlm_size[cls_iter.first];
      cur_lm->setID(new_id);     
      line_lms[cls_iter.first].push_back(cur_lm);
      line_map_nlm_size[cls_iter.first]++;
    }
  }

  for(auto &cls_iter: cur_surf_lms) {
    uint16_t sem_type = cls_iter.first;
    int lm_size = cls_iter.second.size();
    if(lm_size == 0) continue;
    for(auto &cur_lm_iter: cls_iter.second) {
      SurfaceLM::Ptr& cur_lm = cur_lm_iter.second;
      uint32_t new_id = surf_map_size[cls_iter.first] + surf_map_nlm_size[cls_iter.first];
      cur_lm->setID(new_id);     
      surf_lms[cls_iter.first].push_back(cur_lm);  
      surf_map_nlm_size[cls_iter.first]++;
    }
  }

  for(auto &cls_iter: line_lms) {
    for(auto &lm: cls_iter.second) {
      ref_map_->insertLine(lm);
    }
  }
  for(auto &cls_iter: surf_lms) {
    for(auto &lm: cls_iter.second) {
      ref_map_->insertSurface(lm);
    }
  }

  const auto& odom_info = cur_map_->getOdomInfo();
  for(const auto& info: odom_info) {
    ref_map_->addOdomInfo(info);
  }
    
  ref_map_->resort();
  ref_map_->initStruct();
  ref_map_->prune(1.414); 
}

void MultiMapMerger::optimizeGBA() {

  printf("[MultiMapMerger][optimizeGBA] Start Global Bundle Adjustment! ...\n");
  ceres::Problem problem;
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();
  ceres::LocalParameterization *line_local_param = new LineLocalParameterization();
  ceres::LocalParameterization *surface_local_param = new SurfaceLocalParameterization();

  auto &keyframes = base_map_->getKeyFrames();
  auto &line_lms = base_map_->getSemLines();
  auto &surface_lms = base_map_->getSemSurfaces();

  for(auto &iter: keyframes) {
    Frame::Ptr frame = iter.second;
    problem.AddParameterBlock(frame->Twb().parameters().data(), 7, pose_local_param);
    if(frame->id() == 0) {
      problem.SetParameterBlockConstant(frame->Twb().parameters().data());
    }
  }

  for(auto &info: base_map_->getOdomInfo()) {
    RelativePoseFactor *rel_pose_factor = new RelativePoseFactor(info.Tij, info.sqrt_info);
      problem.AddResidualBlock(rel_pose_factor, nullptr, 
        info.fi->Twb().parameters().data(), info.fj->Twb().parameters().data());
  }

  // for(auto &loop_info: vec_loop_info_) {
  //   for(auto &info: loop_info) {
  //     RelativePoseFactor *rel_pose_factor = new RelativePoseFactor(info.Tij, sqrt_info);
  //       problem.AddResidualBlock(rel_pose_factor, nullptr, 
  //         info.fi->Twb().parameters().data(), info.fj->Twb().parameters().data());
  //   }
  // }

  for(auto &sem_iter: line_lms) {
    for(auto &lm_iter: sem_iter.second) {
      LineLM::Ptr lm = lm_iter.second;
      problem.AddParameterBlock(lm->parameters().data(), 4, line_local_param);
      auto obs = lm->getAllObs();
      for(auto &ob: obs) {
        Frame::Ptr frame = ob.first;
        LineOB::Ptr feature = ob.second;
        Eigen::Vector3d const point_a = feature->point_a();
        Eigen::Vector3d const point_b = feature->point_b();
        LaserEdge2PFactor *laser_edge_factor = new LaserEdge2PFactor(point_a, point_b, feature->sqrt_info() / 10.0);
        problem.AddResidualBlock(laser_edge_factor, huber_loss, frame->Twb().parameters().data(), lm->parameters().data());
      }
    }
  }
  for(auto &sem_iter: surface_lms) {
    for(auto &lm_iter: sem_iter.second) {
      SurfaceLM::Ptr lm = lm_iter.second;
      problem.AddParameterBlock(lm->parameters().data(), 3, surface_local_param);
      auto obs = lm->getAllObs();
      for(auto &ob: obs) {
        Frame::Ptr frame = ob.first;
        SurfaceOB::Ptr feature = ob.second;
        std::vector<Eigen::Vector3d> const vertices = feature->vertices();
        // LaserSurf4PFactor *laser_surf_factor = new LaserSurf4PFactor(vertices, feature->sqrt_info() / 10.0);
        LaserSurf3PFactor *laser_surf_factor = new LaserSurf3PFactor(vertices, feature->sqrt_info() / 10.0);
        problem.AddResidualBlock(laser_surf_factor, huber_loss, frame->Twb().parameters().data(), lm->parameters().data());
      }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 50;
  options.function_tolerance = 1e-3;
  options.gradient_tolerance = 1e-3;
  options.initial_trust_region_radius = 1e3;
  options.num_threads = 8;
  // options.max_num_line_search_step_size_iterations = 5;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;

  for(auto &sem_iter: line_lms) {
    for(auto &lm_iter: sem_iter.second) {
      lm_iter.second->double2vector();
    }
  }

  for(auto &sem_iter: surface_lms) {
    for(auto &lm_iter: sem_iter.second) {
      lm_iter.second->double2vector();
    }
  }
}

Transform MultiMapMerger::solveAveragePose(const std::vector<OverlapInfo>& ov_info) {
  Eigen::Quaterniond Qv;
  Eigen::Vector3d Pv{Eigen::Vector3d::Zero()};
  Eigen::Matrix4d A{Eigen::Matrix4d::Zero()};
  for(auto info: ov_info) {
    Eigen::Quaterniond q = info.Trc.q();
    A += q.coeffs() * q.coeffs().transpose();
    Pv += info.Trc.p();
  }
  A /= ov_info.size();
  Pv /= ov_info.size();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> saes(A);
  auto qcoeffs = saes.eigenvectors().rightCols<1>().normalized();
  Qv = Eigen::Quaterniond(qcoeffs(3), qcoeffs(0), qcoeffs(1), qcoeffs(2));
  return Transform(Pv, Qv);
}

} // namespace SLIM
