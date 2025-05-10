#include "vector_map.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>

namespace SLIM {
  
using namespace boost;

uint32_t LineLM::factory_id_ = 0;
uint32_t SurfaceLM::factory_id_ = 0;
uint32_t Block::factory_id_ = 0;
uint32_t VectorMap::factory_id_ = 0;

bool VectorMap::alignFrame(const Frame::Ptr& frame) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  if(keyframes_.size() == 0) { // first frame, construct landmark from every feature

    auto const &line_obs = frame->line_obs();
    auto const &surface_obs = frame->surface_obs();

    for(auto &ob: line_obs)
      insertNewLine(frame, ob);

    for(auto &ob: surface_obs)
      insertNewSurface(frame, ob);

    keyframes_.insert(std::make_pair(frame->id(), frame));
    std::cout << "VectorMap::keyframes_: " <<  keyframes_.size() << std::endl;

    initStruct();
    // std::cout << "pole_lms_        : " <<  sem_line_lms_[POLE_ID].size() << std::endl;
    // std::cout << "traffic_sign_lms_: " <<  sem_surf_lms_[TRAFFIC_SIGN_ID].size() << std::endl;
    // std::cout << "road_lms_        : " <<  sem_surf_lms_[ROAD_ID].size() << std::endl;
    // std::cout << "slidewalk_lms_   : " <<  sem_surf_lms_[SIDEWALK_ID].size() << std::endl;
    // std::cout << "building_lms_    : " <<  sem_surf_lms_[BUILDING_ID].size() << std::endl;
    last_kf_ = frame;
    pivot_kf_ = frame;
    pivot_sqrt_info_ = Eigen::Matrix<double, 6, 6> ::Identity() * 1e8;
    return true;
  }
  else { // associate current frame to map
    AssociateLine(frame);
    AssociateSurface(frame);

    keyframes_.insert(std::make_pair(frame->id(), frame));
    initStruct();
    // std::cout << "pole_lms_        : " <<  sem_line_lms_[POLE_ID].size() << std::endl;
    // std::cout << "traffic_sign_lms_: " <<  sem_surf_lms_[TRAFFIC_SIGN_ID].size() << std::endl;
    // std::cout << "road_lms_        : " <<  sem_surf_lms_[ROAD_ID].size() << std::endl;
    // std::cout << "slidewalk_lms_   : " <<  sem_surf_lms_[SIDEWALK_ID].size() << std::endl;
    // std::cout << "building_lms_    : " <<  sem_surf_lms_[BUILDING_ID].size() << std::endl;
    last_kf_ = frame;
    return true;
  }
}

void VectorMap::releaseAll() {
  keyframes_.clear();
  sem_line_lms_.clear();
  sem_surf_lms_.clear();

  sem_line_ct_struct_.clear();
  sem_surf_ct_struct_.clear();
  odom_info_.clear();
}

void VectorMap::saveJsonFile(const std::string& file) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  Json::Value root;
  root["name"] = Json::Value("SLIM");
  
  for(auto frame_iter: keyframes_) {
    Json::Value frame = writeFrameObject(frame_iter.second);
    root["frames"].append(frame);
  }

  for(auto &cls_iter: sem_line_lms_) {
    uint16_t cls = cls_iter.first;
    auto lms = cls_iter.second;
    for(auto &lm_iter: lms) {
      Json::Value lm = writeLineObject(lm_iter.second);
      root["llms"].append(lm);
    }
  }

  for(auto &cls_iter: sem_surf_lms_) {
    uint16_t cls = cls_iter.first;
    auto lms = cls_iter.second;
    for(auto &lm_iter: lms) {
      Json::Value lm = writeSurfObject(lm_iter.second);
      
      root["slms"].append(lm);
    }
  }
  root["pivot_kf_id"] = pivot_kf_->id();
  for(int r = 0; r < 6; ++r)
    for(int c = 0; c < 6; ++c)
      root["pivot_sqinfo"].append(pivot_sqrt_info_(r, c));

  root["odom_info"] = writeOdomInfo();

  // write 
  std::ofstream os;
  os.open(file, std::ios::out);
  if(!os.is_open()) {
    std::runtime_error("Error: Save VectorMap Failed...");
    return;
  }
  Json::StyledWriter sw;
  os << sw.write(root);
  os.close();
}

void VectorMap::loadJsonFile(const std::string& file) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  TicToc timer;
  printf("[VectorMap] loadJsonFile %s ...", file.c_str());
  std::ifstream is(file, std::ios::binary);
  if(!is.is_open()) {
    std::runtime_error("Error: Read Map File Failed...");
    return;
  }
  Json::Reader reader;
  Json::Value root;
  if(reader.parse(is, root)) {
    std::string name = root["name"].asString();
    auto frames = root["frames"];
    auto llms = root["llms"];
    auto slms = root["slms"];
    auto pivot_kf_id = root["pivot_kf_id"].asUInt();

    for(auto f: frames) {
      Frame::Ptr frame = loadFrame(f);
      keyframes_.insert(std::make_pair(frame->id(), frame));
    }
    for(auto lm: llms) {
      LineLM::Ptr landmark = loadLineObject(lm);
      sem_line_lms_[landmark->semantic_type()].insert(std::make_pair(landmark->id(), landmark));
    }
    for(auto lm: slms) {
      SurfaceLM::Ptr landmark = loadSurfObject(lm);
      sem_surf_lms_[landmark->semantic_type()].insert(std::make_pair(landmark->id(), landmark));
    }
    
    loadOdomInfo(root);
    pivot_kf_ = keyframes_[pivot_kf_id];
    Json::Value pivot_sqinfo = root["pivot_sqinfo"];
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < 6; j++) {
        pivot_sqrt_info_(i, j) = pivot_sqinfo[i*6+j].asDouble();
      }
    }
    // std::cout << "pole_lms_        : " <<  sem_line_lms_[POLE_ID].size() << std::endl;
    // std::cout << "traffic_sign_lms_: " <<  sem_surf_lms_[TRAFFIC_SIGN_ID].size() << std::endl;
    // std::cout << "road_lms_        : " <<  sem_surf_lms_[ROAD_ID].size() << std::endl;
    // std::cout << "slidewalk_lms_   : " <<  sem_surf_lms_[SIDEWALK_ID].size() << std::endl;
    // std::cout << "building_lms_    : " <<  sem_surf_lms_[BUILDING_ID].size() << std::endl;
  }
  else {
    std::runtime_error("Error: The map is empty ...");
    return;
  }
  is.close();
  initStruct();  
  printf(" OK. Time cost: %lf\n", timer.toc());
}

void VectorMap::loadLocalizationMap(const std::string &file) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  TicToc timer;
  printf("[VectorMap] loadJsonFile %s ...", file.c_str());
  std::ifstream is(file, std::ios::binary);
  if(!is.is_open()) {
    std::runtime_error("Error: Read Map File Failed...");
    return;
  }
  Json::Reader reader;
  Json::Value root;
  if(reader.parse(is, root)) {
    std::string name = root["name"].asString();
    // auto frames = root["frames"];
    auto llms = root["llms"];
    auto slms = root["slms"];

    for(auto lm: llms) {
      LineLM::Ptr landmark = loadLineObject(lm, true);
      sem_line_lms_[landmark->semantic_type()].insert(std::make_pair(landmark->id(), landmark));
    }
    for(auto lm: slms) {
      SurfaceLM::Ptr landmark = loadSurfObject(lm, true);
      sem_surf_lms_[landmark->semantic_type()].insert(std::make_pair(landmark->id(), landmark));
    }
    
    // std::map<uint64_t, Frame::Ptr> trajectory;
    // for(auto &f_iter: keyframes_) {
    //   trajectory.insert(std::make_pair(f_iter.second->timestamp(), f_iter.second));
    // }
    // for(auto f_iter = trajectory.begin(); f_iter != std::prev(trajectory.end()); ++f_iter) {
    //   Frame::Ptr fk = f_iter->second;
    //   Frame::Ptr fl = std::next(f_iter)->second;
    //   Transform Tkl = fk->Twb().inverse() * fl->Twb();
    //   odom_info_.push_back(RelPoseInfo(fk, fl, Tkl));
    // }
    // std::cout << "pole_lms_        : " <<  sem_line_lms_[POLE_ID].size() << std::endl;
    // std::cout << "traffic_sign_lms_: " <<  sem_surf_lms_[TRAFFIC_SIGN_ID].size() << std::endl;
    // std::cout << "road_lms_        : " <<  sem_surf_lms_[ROAD_ID].size() << std::endl;
    // std::cout << "slidewalk_lms_   : " <<  sem_surf_lms_[SIDEWALK_ID].size() << std::endl;
    // std::cout << "building_lms_    : " <<  sem_surf_lms_[BUILDING_ID].size() << std::endl;
  }
  else {
    std::runtime_error("Error: The map is empty ...");
    return;
  }
  is.close();
  initStruct();  
  printf(" OK. Time cost: %lf\n", timer.toc());
}

void VectorMap::transform(const Transform& Tsd) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  for(auto &f_iter: keyframes_) {
    uint32_t const fid = f_iter.first;
    Frame::Ptr const frame = f_iter.second;
    Transform const Twb = frame->Twb();
    frame->setTwb(Tsd * Twb);
  }

  for(auto &cls_iter: sem_line_lms_) {
    uint16_t cls = cls_iter.first;
    auto lms = cls_iter.second;
    for(auto &lm_iter: lms) {
      lm_iter.second->transform(Tsd);
    }
  }

  for(auto &cls_iter: sem_surf_lms_) {
    uint16_t cls = cls_iter.first;
    auto lms = cls_iter.second;
    for(auto &lm_iter: lms) {
      lm_iter.second->transform(Tsd);
    }
  }

  initStruct();
}

void VectorMap::resolveLM() {
  for(auto &cls_iter: sem_line_lms_) {
    uint16_t cls = cls_iter.first;
    auto &lms = cls_iter.second;
    for(auto &lm_iter: lms) {
      lm_iter.second->solveByObs();
    }
  }
  for(auto &cls_iter: sem_surf_lms_) {
    uint16_t cls = cls_iter.first;
    auto &lms = cls_iter.second;
    for(auto &lm_iter: lms) {
      lm_iter.second->solveByObs();
    }
  }
}

void VectorMap::resort() {

  uint32_t sorted_id = 0;
  std::map<uint32_t, Frame::Ptr> fmap(keyframes_.begin(), keyframes_.end());
  FrameHashMap fhmap;
  keyframes_.clear();
  for(auto &f_iter: fmap) {
    f_iter.second->setID(sorted_id);
    keyframes_.insert(std::make_pair(sorted_id, f_iter.second));
    sorted_id++;
  }
  // keyframes_ = fhmap;

  std::map<uint16_t, uint32_t> line_idmap;
  std::map<uint16_t, uint32_t> surf_idmap;
  
  for(auto &cls_iter: sem_line_lms_) {
    line_idmap[cls_iter.first] = 0;
  }
  for(auto &cls_iter: sem_surf_lms_) {
    surf_idmap[cls_iter.first] = 0;
  }

  for(auto &cls_iter: sem_line_lms_) {
    uint16_t cls = cls_iter.first;    
    std::map<uint32_t, LineLM::Ptr> lmap(cls_iter.second.begin(), cls_iter.second.end());
    cls_iter.second.clear();
    for(auto lm_iter = lmap.begin(); lm_iter != lmap.end(); lm_iter++) {
      const uint32_t id = line_idmap[cls_iter.first];
      lm_iter->second->setID(id); 
      cls_iter.second.insert(std::make_pair(id, lm_iter->second));
      line_idmap[cls_iter.first]++;
    }
  }

  for(auto &cls_iter: sem_surf_lms_) {
    uint16_t cls = cls_iter.first;    
    std::map<uint32_t, SurfaceLM::Ptr> smap(cls_iter.second.begin(), cls_iter.second.end());
    cls_iter.second.clear();
    for(auto lm_iter = smap.begin(); lm_iter != smap.end(); lm_iter++) {
      const uint32_t id = surf_idmap[cls_iter.first];
      lm_iter->second->setID(id); 
      cls_iter.second.insert(std::make_pair(id, lm_iter->second));
      surf_idmap[cls_iter.first]++;
    }
  }
}


void VectorMap::divideBlocks(const std::map<uint32_t, Frame::Ptr>& pivots, const float range) {
  TicToc timer;
  blocks_.clear();
  for(auto p_iter: pivots) {
    Block::Ptr block(new Block(p_iter.first));
    block->setHostFrame(p_iter.second);
    blocks_.insert(std::make_pair(p_iter.first, block));
  }
  for(auto &block_iter: blocks_) {
    Eigen::Vector3d const pos = block_iter.second->getHostFrame()->Twb().p();
    pcl::PointXYZL index;
    index.getVector3fMap() = pos.cast<float>();
    for(auto &cls_iter: sem_line_ct_struct_) {
      std::vector<int> indices;
      std::vector<float> distances;
      if(cls_iter.second.first->empty()) continue;
      cls_iter.second.second->radiusSearch(index, range, indices, distances);
      for(int i = 0; i < indices.size(); ++i) {
        uint32_t lm_id = cls_iter.second.first->points[indices[i]].label;
        block_iter.second->insertLine(sem_line_lms_[cls_iter.first][lm_id]);
      }
    }
    for(auto &cls_iter: sem_surf_ct_struct_) {
      std::vector<int> indices;
      std::vector<float> distances;
      if(cls_iter.second.first->empty()) continue;
      cls_iter.second.second->radiusSearch(index, range, indices, distances);
      for(int i = 0; i < indices.size(); ++i) {
        uint32_t lm_id = cls_iter.second.first->points[indices[i]].label;
        block_iter.second->insertSurface(sem_surf_lms_[cls_iter.first][lm_id]);
      }
    }
    block_iter.second->initStruct();
  }
  printf("[divideBlocks] Time: %lf ms\n", timer.toc());
}

void VectorMap::divideBlocks(const float density, const float range) {
  TicToc timer;
  pcl::PointCloud<pcl::PointXYZL>::Ptr particles(new pcl::PointCloud<pcl::PointXYZL>());
  for(auto &f_iter: keyframes_) {
    uint32_t id = f_iter.second->id();
    Eigen::Vector3d Pwb = f_iter.second->Twb().p();
    pcl::PointXYZL particle;
    particle.getVector3fMap() = Pwb.cast<float>();
    particle.label = id;
    particles->push_back(particle);
  }
  pcl::VoxelGrid<pcl::PointXYZL> vf;
  vf.setLeafSize(density, density, density);
  vf.setInputCloud(particles);
  vf.filter(*particles);

  blocks_.clear();
  for(auto particle: particles->points) {
    uint32_t const id = particle.label;
    Frame::Ptr const frame = keyframes_[id];
    Block::Ptr block(new Block(id));
    block->setHostFrame(frame);
    blocks_.insert(std::make_pair(id, block));
  }
  for(auto &block_iter: blocks_) {
    Eigen::Vector3d const pos = block_iter.second->getHostFrame()->Twb().p();
    pcl::PointXYZL index;
    index.getVector3fMap() = pos.cast<float>();
    for(auto &cls_iter: sem_line_ct_struct_) {
      std::vector<int> indices;
      std::vector<float> distances;
      if(cls_iter.second.first->empty()) continue;
      cls_iter.second.second->radiusSearch(index, range, indices, distances);
      for(int i = 0; i < indices.size(); ++i) {
        uint32_t lm_id = cls_iter.second.first->points[indices[i]].label;
        block_iter.second->insertLine(sem_line_lms_[cls_iter.first][lm_id]);
      }
    }
    for(auto &cls_iter: sem_surf_ct_struct_) {
      std::vector<int> indices;
      std::vector<float> distances;
      if(cls_iter.second.first->empty()) continue;
      cls_iter.second.second->radiusSearch(index, range, indices, distances);
      for(int i = 0; i < indices.size(); ++i) {
        uint32_t lm_id = cls_iter.second.first->points[indices[i]].label;
        block_iter.second->insertSurface(sem_surf_lms_[cls_iter.first][lm_id]);
      }
    }
    block_iter.second->initStruct();
  }
  // printf("[divideBlocks] Time: %lf ms\n", timer.toc());
}

void VectorMap::extractPivots(const float size) {
  TicToc timer;
  std::map<uint64_t, Frame::Ptr> trajectory;
  for(auto &f_iter: keyframes_) {
    trajectory.insert(std::make_pair(f_iter.second->timestamp(), f_iter.second));
  }
  double distance = 0.0;
  int block_id = 0;
  std::map<uint64_t, Frame::Ptr>::iterator f_iter = trajectory.begin();
  pivots_.insert(std::make_pair(block_id, f_iter->second));
  block_id++;

  Transform last_Twb = f_iter->second->Twb();
  Transform Twb;
  f_iter++;

  for(; f_iter != trajectory.end(); f_iter++) {
    Twb = f_iter->second->Twb();
    double rel_dist = (last_Twb.inverse() * Twb).p().norm();
    if(rel_dist > size) {
      pivots_.insert(std::make_pair(block_id, f_iter->second));
      block_id++;
      last_Twb = f_iter->second->Twb();
      distance = 0.0;
    }
  }
  std::cout << "pivot size: " << pivots_.size() << std::endl;
  printf("[extractPivots] Time: %lf ms\n", timer.toc());
}


void VectorMap::findMatchPivots(const VectorMap::Ptr& other, std::vector<std::pair<uint32_t, uint32_t>>& matches) {
  NodeQuery other_trajectory_query;
  other_trajectory_query.first = CentroidCloudPtr(new CentroidCloud());
  other_trajectory_query.second = CentroidKdTreePtr(new CentroidKdTree());

  std::map<uint32_t, Frame::Ptr>& other_pivots = other->getPivots();

  // construct a map of trajectory (including all keyframes) (sorted by id/timestamp)
  std::map<uint32_t, Frame::Ptr> other_trajectory;
  for(auto &f_iter: other->getKeyFrames()) {
    other_trajectory.insert(std::make_pair(f_iter.second->id(), f_iter.second));
  }

  // build a query of pivots of other map's trajectory
  for(auto iter: other_trajectory) {
    Frame::Ptr frame = iter.second;
    Eigen::Vector3d translation = frame->Twb().p();
    pcl::PointXYZL position;
    position.getVector3fMap() = translation.cast<float>();
    position.label = iter.first;
    other_trajectory_query.first->push_back(position);
  }
  other_trajectory_query.second->setInputCloud(other_trajectory_query.first);

  std::map<uint32_t, Frame::Ptr> match_pivots;
  match_pivots.clear();
  matches.clear();
  std::map<uint32_t, uint32_t> reverse_query;

  int match_pivot_id = 0;
  for(auto iter: pivots_) {
    int this_id = iter.first;
    Eigen::Vector3d translation = iter.second->Twb().p();
    pcl::PointXYZL index;
    index.getVector3fMap() = translation.cast<float>();
    std::vector<int> indices;
    std::vector<float> distances;
    other_trajectory_query.second->nearestKSearch(index, 1, indices, distances);
    pcl::PointXYZL candidate = other_trajectory_query.first->points[indices[0]];
    if(distances[0] < 10.0) {
      Frame::Ptr other_frame = other_trajectory[candidate.label];
      match_pivots.insert(std::make_pair(match_pivot_id, other_frame));
      reverse_query.insert(std::make_pair(match_pivot_id, this_id));
      matches.push_back(std::make_pair(this_id, match_pivot_id));
      match_pivot_id++;
    }
  }

  other_pivots = match_pivots;
}


void VectorMap::mergePivots(const std::map<uint32_t, Frame::Ptr>& other_pivots) {
  NodeQuery query;
  query.first = CentroidCloudPtr(new CentroidCloud());
  query.second = CentroidKdTreePtr(new CentroidKdTree());

  std::cout << "current pivot size: " << pivots_.size() << std::endl;

  for(auto p_iter: pivots_) {
    uint32_t id = p_iter.first;
    Eigen::Vector3d translation = p_iter.second->Twb().p();
    pcl::PointXYZL pivot;
    pivot.getVector3fMap() = translation.cast<float>();
    pivot.label = id;
    query.first->push_back(pivot);
  }
  query.second->setInputCloud(query.first);

  int pivot_id = pivots_.size();
  for(auto p_iter: other_pivots) {
    uint32_t id = p_iter.first;
    Eigen::Vector3d translation = p_iter.second->Twb().p();
    pcl::PointXYZL pivot;
    pivot.getVector3fMap() = translation.cast<float>();
    pivot.label = id;

    std::vector<int> indices;
    std::vector<float> distances;
    query.second->nearestKSearch(pivot, 1, indices, distances);
    std::cout << "distance: " << distances[0] << std::endl;
    if(distances[0] > 400.0) {
      pivots_.insert(std::make_pair(pivot_id, p_iter.second));
      pivot_id++;
      std::cout << "add pivot" << std::endl;
    }
    else {
      std::cout << "don't add pivot" << std::endl;
    }
  }
  assert(pivot_id == pivots_.size());

  std::cout << "other pivot size: " << other_pivots.size() << std::endl;
  std::cout << "after merge pivot size: " << pivots_.size() << std::endl;
}

void VectorMap::refineRelativePose(VectorMap::Ptr& other,
                                  Transform& Tab, 
                                  std::map<uint16_t, LineMatchVec>& line_matches, 
                                  std::map<uint16_t, SurfMatchVec>& surf_matches) {
  auto other_frames = other->getKeyFrames();
  auto other_sem_line_lms = other->getSemLines();
  auto other_sem_surf_lms = other->getSemSurfaces();

  line_matches.clear();
  surf_matches.clear();

  ceres::Problem problem;
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
  ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  

  problem.AddParameterBlock(Tab.parameters().data(), 7, pose_local_param);

  // for every landmark in <other> map, find the nearest landmark in current map
  for(auto &cls_iter: other_sem_line_lms) {
    int lm_size = cls_iter.second.size();
    auto &pivots = sem_line_ct_struct_[cls_iter.first].first;
    auto &tree = sem_line_ct_struct_[cls_iter.first].second;
    auto &lms = sem_line_lms_[cls_iter.first];

    for(auto &other_lm_iter: cls_iter.second) {
      LineLM::Ptr other_lm = other_lm_iter.second;
      Eigen::Vector3d const& other_c = Tab * other_lm->centroid();

      pcl::PointXYZL index, query;
      index.getVector3fMap() = other_c.cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      tree->nearestKSearch(index, 1, indices, distances);
      query = pivots->points[indices[0]];

      uint32_t const lm_id = query.label;
      auto this_lm_iter = lms.find(lm_id);
      if(this_lm_iter != lms.end()) {
        LineLM::Ptr this_lm = this_lm_iter->second;
        Eigen::Vector3d const& this_c = this_lm->centroid();
        Eigen::Vector3d const& this_n = this_lm->normal();
        double dist = ((Eigen::Matrix3d::Identity() - this_n * this_n.transpose()) * (other_c - this_c)).norm();
        if(dist < 0.5) {
          line_matches[cls_iter.first].push_back(std::make_pair(this_lm, other_lm));
          Eigen::Vector3d const p0 = other_lm->centroid() + other_lm->normal(), p1 = other_lm->centroid() - other_lm->normal();
          LaserPointToLineFactor *f0 = new LaserPointToLineFactor(p0, this_c, this_n, 1/0.2);
          LaserPointToLineFactor *f1 = new LaserPointToLineFactor(p1, this_c, this_n, 1/0.2);
          problem.AddResidualBlock(f0, huber_loss, Tab.parameters().data());
          problem.AddResidualBlock(f1, huber_loss, Tab.parameters().data());
        }   
      }
    }
  }

  for(auto &cls_iter: other_sem_surf_lms) {
    int lm_size = cls_iter.second.size();
    auto &pivots = sem_surf_ct_struct_[cls_iter.first].first;
    auto &tree = sem_surf_ct_struct_[cls_iter.first].second;
    auto &lms = sem_surf_lms_[cls_iter.first];

    for(auto &other_lm_iter: cls_iter.second) {
      SurfaceLM::Ptr other_lm = other_lm_iter.second;
      Eigen::Vector3d const& other_c = Tab * other_lm->centroid();

      pcl::PointXYZL index, query;
      index.getVector3fMap() = other_c.cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      tree->nearestKSearch(index, 1, indices, distances);
      query = pivots->points[indices[0]];

      uint32_t const lm_id = query.label;
      auto this_lm_iter = lms.find(lm_id);
      if(this_lm_iter != lms.end()) {
        SurfaceLM::Ptr this_lm = this_lm_iter->second;
        Eigen::Vector3d const& this_c = this_lm->centroid();
        Eigen::Vector3d const& this_n = this_lm->normal();
        double ndist = (this_n.transpose() * (other_c - this_c)).norm();
        double dist = (other_c - this_c).norm();
        if(ndist < 0.5 && dist < 1.0) {
          surf_matches[cls_iter.first].push_back(std::make_pair(this_lm, other_lm));
          LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(other_lm->centroid(), this_c, this_n, 1/0.2);
          problem.AddResidualBlock(f, huber_loss, Tab.parameters().data());
        }
      }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 10;
  options.function_tolerance = 1e-4;
  options.gradient_tolerance = 1e-4;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << std::endl;
  std::cout << "[refineRelativePose]: Tab:" << std::endl << Tab.matrix() << std::endl;
} 


void VectorMap::visualize(
  std::map<uint32_t, Transform>& poses, std::map<uint32_t, Transform>& gt_poses, std::vector<std::pair<uint32_t, uint32_t>>& rel_pose_infos,
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& poles, std::vector<std::vector<Eigen::Vector3d>>& roads, std::vector<std::vector<Eigen::Vector3d>>& buildings,
  std::vector<std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>>& pole_obs, std::vector<std::vector<std::vector<Eigen::Vector3d>>>& road_obs, std::vector<std::vector<std::vector<Eigen::Vector3d>>>& building_obs
) {
  
  std::lock_guard<std::mutex> lock(data_mutex_);

  poses.clear();
  gt_poses.clear();
  for(auto &f: keyframes_) {
    poses.insert(std::make_pair(f.first, f.second->Twb()));
    gt_poses.insert(std::make_pair(f.first, f.second->gt_Twb_));
  }

  rel_pose_infos.clear();
  for(auto &info: odom_info_) {
    rel_pose_infos.push_back(std::make_pair(info.fi->id(), info.fj->id()));
  }

  poles.clear();
  for(auto &cls_iter: sem_line_lms_) {
    uint16_t const sem_type = cls_iter.first;
    for(auto &lm_iter: cls_iter.second) {
      LineLM::Ptr& lm = lm_iter.second;
      LineInfo line_info = lm->getLineInfo();
      // lm->solveByObs();
      Eigen::Vector3d const pt = lm->pa();
      Eigen::Vector3d const pb = lm->pb();
      poles.push_back(std::make_pair(pt, pb));
    }
  }
  roads.clear();
  buildings.clear();
  for(auto &cls_iter: sem_surf_lms_) {
    uint16_t const sem_type = cls_iter.first;
    if(sem_type == ROAD_ID) {
      for(auto &lm_iter: cls_iter.second) {
        uint32_t id = lm_iter.first;
        SurfaceLM::Ptr& lm = lm_iter.second; 
        auto vertices = lm->vertices();
        roads.push_back(vertices);
      }
    }
    else if(sem_type == BUILDING_ID) {
      for(auto &lm_iter: cls_iter.second) {
        uint32_t id = lm_iter.first;
        SurfaceLM::Ptr& lm = lm_iter.second; 
        auto vertices = lm->vertices();
        buildings.push_back(vertices);
      }
    }
  }
}



void VectorMap::filter(const int threshold) {
  std::lock_guard<std::mutex> lock(data_mutex_);

  // Step1: merge similar landmarks
  std::set<uint32_t> sem_line_lm_to_be_merged;
  std::set<uint32_t> sem_surf_lm_to_be_merged;

  // for(auto &cls_iter: sem_surf_lms_) {
  //   uint16_t const sem_type = cls_iter.first;

  //   int lm_size = cls_iter.second.size();
  //   if(lm_size == 0) continue;
  //   CentroidCloudPtr &nodes = sem_surf_ct_struct_[sem_type].first;
  //   CentroidKdTreePtr &tree = sem_surf_ct_struct_[sem_type].second;
  //   if(nodes.get() == nullptr) continue;
  //   if(nodes->empty()) continue;    

  //   for(auto &lm_iter: cls_iter.second) {
  //     uint32_t id = lm_iter.first;
  //     SurfaceLM::Ptr& cur_lm = lm_iter.second;
  //     Eigen::Vector3d const cur_c = cur_lm->centroid();
  //     Eigen::Vector3d const cur_n = cur_lm->normal();
  //     pcl::PointXYZL index;
  //     index.getVector3fMap() = cur_lm->centroid().cast<float>();
  //     std::vector<int> indices;
  //     std::vector<float> distances;
  //     tree->radiusSearch(index, cur_lm->getRadius(), indices, distances);
  //     if(indices.empty()) 
  //       continue;

  //     std::vector<SurfaceLM::Ptr> nb_surf_lms;
  //     for(auto indice: indices) {
  //       uint32_t const lm_id = std::round(nodes->points[indice].intensity);
  //       nb_surf_lms.push_back(sem_surf_lms_[sem_type][lm_id]);        
  //     }

  //     // get the landmark with max radius, if there is another larger plane, return
  //     SurfaceLM::Ptr max_surf = getLargestSurf(nb_surf_lms);
  //     if(max_surf->getRadius() > cur_lm->getRadius())
  //       continue;

  //     std::vector<SurfaceLM::Ptr> inlier = extractConsistentSurf(cur_lm, nb_surf_lms);
  //     std::vector<uint32_t> neighbor_set;
  //     for(auto obj: inlier) {
  //       if(sem_surf_lm_to_be_merged.find(obj->id()) != sem_surf_lm_to_be_merged.end())
  //         continue;
  //       neighbor_set.push_back(obj->id());
  //     }
  //     if(!neighbor_set.empty()) {
  //       sem_surf_lm_pair_to_merge[sem_type].push_back(std::make_pair(id, neighbor_set));
  //     }
  //   }
  // }






  // Step2: remove noise
  std::map<uint16_t, std::vector<uint32_t>> sem_line_lm_remove;
  std::map<uint16_t, std::vector<uint32_t>> sem_surf_lm_remove;
  int sem_line_remove_cnt = 0;
  int sem_surf_remove_cnt = 0;

  for(auto &cls_iter: sem_line_lms_) {
    uint16_t const sem_type = cls_iter.first;
    for(auto &lm_iter: cls_iter.second) {
      uint32_t id = lm_iter.first;
      if(lm_iter.second->getObSize() < threshold) {
        sem_line_lm_remove[sem_type].push_back(id);
      }
    }
  }

  for(auto &cls_iter: sem_surf_lms_) {
    uint16_t const sem_type = cls_iter.first;
    for(auto &lm_iter: cls_iter.second) {
      uint32_t id = lm_iter.first;
      if(lm_iter.second->getObSize() < threshold) {
        sem_surf_lm_remove[sem_type].push_back(id);
      }
    }
  }

  for(auto &cls_iter: sem_line_lm_remove) {
    for(auto id: cls_iter.second) {
      sem_line_lms_[cls_iter.first].erase(id);
      sem_line_remove_cnt++;
    }
  }
  for(auto &cls_iter: sem_surf_lm_remove) {
    for(auto id: cls_iter.second) {
      sem_surf_lms_[cls_iter.first].erase(id);
      sem_surf_remove_cnt++;
    }
  }
  printf("[VectorMap] Filter Map, Remove Line: %d, Remove Surface: %d\n", sem_line_remove_cnt, sem_surf_remove_cnt);

  resort();
  initStruct();
}


// initial version (semantic mapping) of prune
// void VectorMap::prune() {

//   std::lock_guard<std::mutex> lock(data_mutex_);

//   std::map<LineLM::Ptr, std::vector<LineLM::Ptr>> line_merge_map;
//   std::map<SurfaceLM::Ptr, std::vector<SurfaceLM::Ptr>> surf_merge_map;
//   std::map<uint16_t, std::set<uint32_t>> sem_line_lm_to_update;
//   std::map<uint16_t, std::set<uint32_t>> sem_surf_lm_to_update;
//   std::map<uint16_t, std::set<uint32_t>> sem_line_lm_to_be_merged;
//   std::map<uint16_t, std::set<uint32_t>> sem_surf_lm_to_be_merged;

//   int line_lm_remove_cnt = 0, surf_lm_remove_cnt = 0;
//   for(auto &cls_iter: sem_line_lms_) {
//     uint16_t const sem_type = cls_iter.first;
//     for(auto &lm_iter: cls_iter.second) {
//       uint32_t id = lm_iter.first;
//       // if the landmark is in to_be_merge set, it will be removed
//       if(sem_line_lm_to_be_merged[sem_type].find(id) != sem_line_lm_to_be_merged[sem_type].end() ||
//          sem_line_lm_to_update[sem_type].find(id) != sem_line_lm_to_update[sem_type].end()) {
//         continue;
//       }
//       int lm_size = cls_iter.second.size();
//       if(lm_size == 0) continue;
//       CentroidCloudPtr &nodes = sem_line_ct_struct_[cls_iter.first].first;
//       if(nodes.get() == nullptr) continue;
//       if(nodes->empty()) continue;
//       CentroidKdTreePtr &tree = sem_line_ct_struct_[cls_iter.first].second;
//       std::unordered_map<uint32_t, LineLM::Ptr> &lms = sem_line_lms_[cls_iter.first];

//       LineLM::Ptr lm = lm_iter.second;
//       Eigen::Vector3d const ci = lm->centroid();
//       Eigen::Vector3d const ni = lm->normal();
//       pcl::PointXYZL index, query;
//       index.getVector3fMap() = ci.cast<float>();
//       std::vector<int> indices;
//       std::vector<float> distances;
      
//       tree->radiusSearch(index, 0.5 * GetSemanticResolution(sem_type), indices, distances);
//       if(indices.size() <= 1)
//         continue;

//       for(int i = 1; i < indices.size(); ++i) {
//         query = nodes->points[indices[i]];
//         uint32_t const nlm_id = std::round(query.intensity);
//         auto nlm_iter = lms.find(nlm_id);
//         if(nlm_iter != lms.end()) {
//           Eigen::Vector3d const cj = nlm_iter->second->centroid();
//           Eigen::Vector3d const nj = nlm_iter->second->normal();
//           Eigen::Matrix3d const nmat = Eigen::Matrix3d::Identity() - ni * ni.transpose();
//           Eigen::Vector3d const r = nmat * (cj - ci);
//           double theta = std::acos(ni.dot(nj));
//           if((theta < M_PI/18 || theta > 17*M_PI/18) && r.norm() < 0.1) {
//             lm->merge(nlm_iter->second);
//             // line_merge_map[lm].push_back(nlm_iter->second);
//             sem_line_lm_to_be_merged[sem_type].insert(nlm_iter->second->id());    
//             sem_line_lm_to_update[sem_type].insert(lm->id());
//             line_lm_remove_cnt++;
//           }
//         }
//       }
//     }
//   }
//   std::cout << "merge line feature" << std::endl;

//   for(auto &cls_iter: sem_surf_lms_) {
//     uint16_t const sem_type = cls_iter.first;
//     for(auto &lm_iter: cls_iter.second) {
//       uint32_t id = lm_iter.first;
//       if(sem_surf_lm_to_be_merged.find(id) != sem_surf_lm_to_be_merged.end() ||
//          sem_surf_lm_to_update.find(id) != sem_surf_lm_to_update.end()) {
//         continue;
//       }

//       int lm_size = cls_iter.second.size();
//       if(lm_size == 0) continue;
//       CentroidCloudPtr &nodes = sem_surf_ct_struct_[cls_iter.first].first;
//       if(nodes.get() == nullptr) continue;
//       if(nodes->empty()) continue;
//       CentroidKdTreePtr &tree = sem_surf_ct_struct_[cls_iter.first].second;
//       std::unordered_map<uint32_t, SurfaceLM::Ptr> &lms = sem_surf_lms_[cls_iter.first];

//       SurfaceLM::Ptr lm = lm_iter.second;
//       Eigen::Vector3d const ci = lm->centroid();
//       Eigen::Vector3d const ni = lm->normal();

//       pcl::PointXYZL index, query;
//       index.getVector3fMap() = ci.cast<float>();
//       std::vector<int> indices;
//       std::vector<float> distances;
//       tree->radiusSearch(index, 0.5 * GetSemanticResolution(sem_type), indices, distances);
//       if(indices.size() == 1)
//         continue;

//       for(int i = 1; i < indices.size(); ++i) {
//         query = nodes->points[indices[i]];
//         uint32_t const nlm_id = std::round(query.intensity);
//         auto nlm_iter = lms.find(nlm_id);

//         if(nlm_iter != sem_surf_lms_[sem_type].end()) {
//           Eigen::Vector3d const cj = nlm_iter->second->centroid();
//           Eigen::Vector3d const nj = nlm_iter->second->normal();
//           SurfaceInfo surface(ci, ni);
//           double const ndist = surface.distance(cj);
//           Eigen::Vector3d const pedal = surface.pedal(cj);
//           double const pdist = (ci - pedal).norm();
//           double const theta = std::acos(ni.dot(nj));
//           if((theta < M_PI/18 || theta > 17*M_PI/18) && std::abs(ndist) < 0.1 && pdist < 0.5 * GetSemanticResolution(sem_type)) {
//             lm->merge(nlm_iter->second);
//             sem_surf_lm_to_be_merged[sem_type].insert(nlm_iter->second->id());
//             sem_surf_lm_to_update[sem_type].insert(lm->id());
//             surf_lm_remove_cnt++;
//           }
//         }
//       }
//     }
//   }

//   std::cout << "merge surf feature" << std::endl;
//   printf("[MapMerger] Prune Map, Remove Line: %d, Remove Surface: %d\n", line_lm_remove_cnt, surf_lm_remove_cnt);
//   // for(auto &lm_iter: line_merge_map) {
//   //   auto lm = lm_iter.first;
//   //   for(auto &mlm_iter: lm_iter.second) {
//   //     lm->merge(mlm_iter);
//   //   }
//   // }

//   // for(auto &lm_iter: surf_merge_map) {
//   //   auto lm = lm_iter.first;
//   //   for(auto &mlm_iter: lm_iter.second) {
//   //     lm->merge(mlm_iter);
//   //   }
//   // }

//   for(auto &cls_iter: sem_line_lm_to_be_merged) {
//     uint16_t sem_type = cls_iter.first;
//     for(auto &id: cls_iter.second) {
//       sem_line_lms_[sem_type].erase(id);
//     }
//     std::cout << "remain landmark: " << sem_line_lms_[sem_type].size() << std::endl;
//   }

//   for(auto &cls_iter: sem_surf_lm_to_be_merged) {
//     uint16_t sem_type = cls_iter.first;
//     for(auto &id: cls_iter.second) {
//       sem_surf_lms_[sem_type].erase(id);
//     }
//     std::cout << "remain landmark: " << sem_surf_lms_[sem_type].size() << std::endl;
//   }

//   resort();
// }


void VectorMap::prune(const float ratio) {

  std::lock_guard<std::mutex> lock(data_mutex_);

  
  for(uint32_t iter = 0; iter < 10; iter++) {

    std::map<uint16_t, std::set<uint32_t>> sem_line_lm_to_update; //  line: {cls, set<ref_id>}
    std::map<uint16_t, std::set<uint32_t>> sem_surf_lm_to_update; //  surface: {cls, set<ref_id>}
    std::map<uint16_t, std::set<uint32_t>> sem_line_lm_to_be_merged; //  line: {cls, set<neighbor_id>}
    std::map<uint16_t, std::set<uint32_t>> sem_surf_lm_to_be_merged; //  surface: {cls, set<neighbor_id>}

    int line_lm_remove_cnt = 0, surf_lm_remove_cnt = 0;
    for(auto &cls_iter: sem_line_lms_) {
      uint16_t const sem_type = cls_iter.first;
      for(auto &lm_iter: cls_iter.second) {
        uint32_t id = lm_iter.first;
        // if the landmark is in to_be_merge or to_be_update set, skip
        if(sem_line_lm_to_be_merged[sem_type].find(id) != sem_line_lm_to_be_merged[sem_type].end() ||
          sem_line_lm_to_update[sem_type].find(id) != sem_line_lm_to_update[sem_type].end()) {
          continue;
        }
        int lm_size = cls_iter.second.size();
        if(lm_size == 0) continue;
        CentroidCloudPtr &nodes = sem_line_ct_struct_[cls_iter.first].first;
        if(nodes.get() == nullptr) continue;
        if(nodes->empty()) continue;
        CentroidKdTreePtr &tree = sem_line_ct_struct_[cls_iter.first].second;
        std::unordered_map<uint32_t, LineLM::Ptr> &lms = sem_line_lms_[cls_iter.first];

        LineLM::Ptr lm = lm_iter.second;
        Eigen::Vector3d const ci = lm->centroid();
        Eigen::Vector3d const ni = lm->normal();
        pcl::PointXYZL index, query;
        index.getVector3fMap() = ci.cast<float>();
        std::vector<int> indices;
        std::vector<float> distances;
        
        // tree->radiusSearch(index, 0.5 * GetSemanticResolution(sem_type), indices, distances);
        tree->radiusSearch(index, 2.0, indices, distances);
        if(indices.size() <= 1)
          continue;

        for(int i = 0; i < indices.size(); ++i) {
          query = nodes->points[indices[i]];
          uint32_t const nlm_id = query.label;
          if(lm->id() == nlm_id)
            continue;   

          auto nlm_iter = lms.find(nlm_id);
          if(nlm_iter != lms.end()) {
            Eigen::Vector3d const cj = nlm_iter->second->centroid();
            Eigen::Vector3d const nj = nlm_iter->second->normal();
            Eigen::Matrix3d const nmat = Eigen::Matrix3d::Identity() - ni * ni.transpose();
            Eigen::Vector3d const r = nmat * (cj - ci);
            double theta = std::acos(ni.dot(nj));
            if((theta < M_PI/18 || theta > 17*M_PI/18) && r.norm() < 0.4) {
              lm->merge(nlm_iter->second);
              // line_merge_map[lm].push_back(nlm_iter->second);
              sem_line_lm_to_be_merged[sem_type].insert(nlm_iter->second->id());    
              sem_line_lm_to_update[sem_type].insert(lm->id());
              line_lm_remove_cnt++;
            }
          }
        }
      }
    }
    std::cout << "merge line feature" << std::endl;

    for(auto &cls_iter: sem_surf_lms_) {
      uint16_t const sem_type = cls_iter.first;

      //     std::map<uint16_t, std::map<double, SurfaceLM::Ptr>> surf_map;
      std::map<double, SurfaceLM::Ptr> ordered_map;
      for(auto &lm_iter: cls_iter.second) {
        ordered_map[lm_iter.second->getRadius()] = lm_iter.second;
      }
  
      // for(auto &lm_iter: cls_iter.second) {
      //   uint32_t id = lm_iter.first;
      for (auto lm_iter = ordered_map.rbegin(); lm_iter != ordered_map.rend(); ++lm_iter) {
        SurfaceLM::Ptr lm = lm_iter->second;
        uint32_t id = lm->id();
        // if the landmark is in to_be_merge or to_be_update set, skip
        if(sem_surf_lm_to_be_merged[sem_type].find(id) != sem_surf_lm_to_be_merged[sem_type].end() ||
          sem_surf_lm_to_update[sem_type].find(id) != sem_surf_lm_to_update[sem_type].end()) {
          continue;
        }

        int lm_size = cls_iter.second.size();
        if(lm_size == 0) continue;
        CentroidCloudPtr &nodes = sem_surf_ct_struct_[cls_iter.first].first;
        if(nodes.get() == nullptr) continue;
        if(nodes->empty()) continue;
        CentroidKdTreePtr &tree = sem_surf_ct_struct_[cls_iter.first].second;
        std::unordered_map<uint32_t, SurfaceLM::Ptr> &lms = sem_surf_lms_[cls_iter.first];
        Eigen::Vector3d const ci = lm->centroid();
        Eigen::Vector3d const ni = lm->normal();

        pcl::PointXYZL index, query;
        index.getVector3fMap() = ci.cast<float>();
        std::vector<int> indices;
        std::vector<float> distances;
        // tree->radiusSearch(index, lm->getRadius() * std::sqrt(2), indices, distances);
        tree->radiusSearch(index, lm->getRadius() * ratio, indices, distances);
        if(indices.size() <= 1)
          continue;

        std::vector<SurfaceLM::Ptr> inliers;
        for(int i = 0; i < indices.size(); ++i) {
          query = nodes->points[indices[i]];
          uint32_t const nlm_id = query.label;
          if(lm->id() == nlm_id)
            continue;

          auto nlm_iter = lms.find(nlm_id);

          if(nlm_iter != sem_surf_lms_[sem_type].end()) {
            SurfaceLM::Ptr nlm = nlm_iter->second;

            if(sem_surf_lm_to_be_merged[sem_type].find(nlm->id()) != sem_surf_lm_to_be_merged[sem_type].end() ||
              sem_surf_lm_to_update[sem_type].find(nlm->id()) != sem_surf_lm_to_update[sem_type].end()) {
              continue;
            }

            Eigen::Vector3d const cj = nlm->centroid();
            Eigen::Vector3d const nj = nlm->normal();
            SurfaceInfo surface(ci, ni);
            double const ndist = surface.distance(cj);
            Eigen::Vector3d const pedal = surface.pedal(cj);
            double const pdist = (ci - pedal).norm();
            double const theta = std::acos(ni.dot(nj));
            if((theta < M_PI/18 || theta > 17*M_PI/18) && std::abs(ndist) < 0.3) {
              inliers.push_back(nlm);
              // lm->merge(nlm);
              // sem_surf_lm_to_be_merged[sem_type].insert(nlm->id());
              // sem_surf_lm_to_update[sem_type].insert(lm->id());
              // surf_lm_remove_cnt++;
            }
          }
        }
        if(inliers.size() >= 1) {
          sem_surf_lm_to_update[sem_type].insert(lm->id());
          for(auto inlier_iter: inliers) {
            sem_surf_lm_to_be_merged[sem_type].insert(inlier_iter->id());
            lm->merge(inlier_iter);
            surf_lm_remove_cnt++;
          }
        }
      }
    }
    printf("[MapMerger] Prune Map, Remove Line: %d, Remove Surface: %d\n", line_lm_remove_cnt, surf_lm_remove_cnt);

    if(line_lm_remove_cnt == 0 && surf_lm_remove_cnt == 0)
      break;

    for(auto &cls_iter: sem_line_lm_to_be_merged) {
      uint16_t sem_type = cls_iter.first;
      for(auto &id: cls_iter.second) {
        sem_line_lms_[sem_type].erase(id);
      }
      std::cout << "remain landmark: " << sem_line_lms_[sem_type].size() << std::endl;
    }

    for(auto &cls_iter: sem_surf_lm_to_be_merged) {
      uint16_t sem_type = cls_iter.first;

      for(auto &id: cls_iter.second) {
        if(sem_surf_lm_to_update[sem_type].find(id) != sem_surf_lm_to_update[sem_type].end()) {
          printf("Warning!\n");
          continue;
        }        
        sem_surf_lms_[sem_type].erase(id);
      }
      std::cout << "remain landmark: " << sem_surf_lms_[sem_type].size() << std::endl;
    }
    // resolveLM();
    resort();
    initStruct();
    resolveLM();
  }
  printf("Prune Done!\n");
}


void VectorMap::extractSparseStruct() {

  TicToc timer;
  const double decay_factor = 0.1;
  std::map<uint32_t, Frame::Ptr> retain_frames;
  std::map<uint32_t, Frame::Ptr> marg_frames;
  pcl::PointCloud<pcl::PointXYZL>::Ptr nodes(new pcl::PointCloud<pcl::PointXYZL>());
  pcl::PointCloud<pcl::PointXYZL>::Ptr sparse_nodes(new pcl::PointCloud<pcl::PointXYZL>());
  for(auto &iter: keyframes_) {
    Frame::Ptr frame = iter.second;
    pcl::PointXYZL node;
    node.getVector3fMap() = (frame->constTwb().p()).cast<float>();
    node.label = frame->id();
    nodes->push_back(node);
  }
  pcl::VoxelGrid<pcl::PointXYZL> vf;
  vf.setLeafSize(10.0, 10.0, 10.0);
  vf.setInputCloud(nodes);
  vf.filter(*sparse_nodes);

  pcl::search::KdTree<pcl::PointXYZL> sparse_kdtree;
  sparse_kdtree.setInputCloud(sparse_nodes);

  for(auto node: sparse_nodes->points) {
    uint32_t id = node.label;
    Frame::Ptr frame = keyframes_[id];
    retain_frames.insert(std::make_pair(id, frame));
  }

  for(auto &iter: keyframes_) {
    Frame::Ptr frame = iter.second;
    if(retain_frames.find(iter.first) == retain_frames.end()) {
      marg_frames.insert(std::make_pair(iter.first, frame));
    }
  }
  std::cout << "Retain frame: " << retain_frames.size() << std::endl;
  std::cout << "Marg frame: " << marg_frames.size() << std::endl;
  std::cout << "All frame: " << keyframes_.size() << std::endl;
  std::cout << "RelPoseInfo frame: " << odom_info_.size() << std::endl;

  NFRSolver nfr_solver;
  for(auto &frame_iter: retain_frames) {
    nfr_solver.addRetainFrame(frame_iter.second);    
  }
  for(auto &frame_iter: marg_frames) {
    nfr_solver.addMargFrame(frame_iter.second);    
  }
  nfr_solver.setPivotFrame(pivot_kf_, pivot_sqrt_info_);

  for(auto cls_iter: sem_line_lms_) {
    for(auto lm_iter: cls_iter.second) {
      LineLM::Ptr lm_ptr = lm_iter.second;
      nfr_solver.addRetainLine(lm_ptr);
    }
  }
  for(auto cls_iter: sem_surf_lms_) {
    for(auto lm_iter: cls_iter.second) {
      SurfaceLM::Ptr lm_ptr = lm_iter.second;
      nfr_solver.addRetainSurf(lm_ptr);
    }
  }
  
  for(auto info: odom_info_) {
    nfr_solver.addRelPoseInfo(info);
  }

  printf("[VectorMap] Sparsify the keyframes! \n");
  nfr_solver.init();
  // nfr_solver.buildDenseStruct();
  nfr_solver.buildSparseStruct();
  printf("[VectorMap] NFR is solved successfully! \n");
  printf("[VectorMap] Start to recover factors! \n");
  // nfr_solver.checkAccuracy();

  printf("[extractSparseStruct] Add Line NF ...\n");
  for(auto &sem_iter: sem_line_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      LineLM::Ptr& lm = lm_iter.second;
      pcl::PointXYZL lm_pos;
      lm_pos.getVector3fMap() = lm->centroid().cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      sparse_kdtree.nearestKSearch(lm_pos, 1, indices, distances);
      Frame::Ptr f = keyframes_[sparse_nodes->points[indices[0]].label];
      nfr_solver.addFrameToLineNF(f, lm);
    }
  }
  printf("[extractSparseStruct] Add Surf NF ...\n");
  for(auto &sem_iter: sem_surf_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      SurfaceLM::Ptr& lm = lm_iter.second;
      pcl::PointXYZL lm_pos;
      lm_pos.getVector3fMap() = lm->centroid().cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      sparse_kdtree.nearestKSearch(lm_pos, 1, indices, distances);
      Frame::Ptr f = keyframes_[sparse_nodes->points[indices[0]].label];
      nfr_solver.addFrameToSurfNF(f, lm);
    }
  }

  printf("[extractSparseStruct] Extract MST ...\n");
  std::vector<Frame::Ptr> frame_vec;
  for(auto kf_iter: retain_frames) {
    frame_vec.push_back(kf_iter.second);
  }

  typedef adjacency_list<vecS, vecS, undirectedS,
      no_property, property<edge_weight_t, double>> Graph;
  typedef graph_traits<Graph>::edge_descriptor Edge;
  typedef graph_traits<Graph>::vertex_descriptor Vertex;

  Graph g;
  std::vector<Vertex> vertices;
  for(int i = 0; i < frame_vec.size(); i++) {
    Vertex v = add_vertex(g);
    vertices.push_back(v);
  }

  for(int i = 0; i < frame_vec.size(); i++) {
    for(int j = i+1; j < frame_vec.size(); j++) {
      Eigen::Vector3d p0 = frame_vec[i]->Twb().p();
      Eigen::Vector3d p1 = frame_vec[j]->Twb().p();
      double dist = (p0 - p1).norm();
      if(dist < 100.0) {
        Edge e;
        bool success;
        tie(e, success) = add_edge(i, j, dist, g);
      }
    }
  }

  std::vector<boost::graph_traits<Graph>::vertex_descriptor> p(boost::num_vertices(g));
  boost::prim_minimum_spanning_tree(g, &p[0]);
  std::vector<std::pair<int, int>> mst_edges;
  for (std::size_t i = 0; i != p.size(); ++i) {
    if (p[i] != i) {
      mst_edges.push_back(std::make_pair(p[i], i));
    }
  }

  if(mst_edges.size() != retain_frames.size() - 1) {
    printf("Edges: %ld, Frames: %ld\n", mst_edges.size(), retain_frames.size());
    throw std::runtime_error("Invalid mst edge size!");
  }

  for(auto iter: mst_edges) {
    nfr_solver.addFrameToFrameNF(frame_vec[iter.first], frame_vec[iter.second]);
  }

  printf("[extractSparseStruct] Add Prior NF ...\n");
  nfr_solver.addFramePriorNF(pivot_kf_);
  printf("[extractSparseStruct] Solve NFR Close-Form ...\n");
  nfr_solver.solveCF();

  uint32_t line_id = 0;
  for(auto &factor: nfr_solver.line_factors_) {
    LineLM::Ptr& lm = factor.lm_ptr;
    lm->resetObs();
    LineOB::Ptr ob = LineOB::Ptr(new LineOB(lm->semantic_type(), factor.ob[0], factor.ob[1], factor.point_num));
    ob->setSqrtInfo(factor.sqrt_info);
    lm->addNewOb(factor.frame_ptr, ob);
    lm->solveByObs();
    line_id++;
  }

  uint32_t surf_id = 0;
  for(auto &factor: nfr_solver.surf_factors_) {
    SurfaceLM::Ptr& lm = factor.lm_ptr;
    lm->resetObs();
    SurfaceOB::Ptr ob = SurfaceOB::Ptr(new SurfaceOB(lm->semantic_type(), factor.ob, factor.point_num));
    ob->setSqrtInfo(factor.sqrt_info);
    lm->addNewOb(factor.frame_ptr, ob);
    lm->solveByObs();
    surf_id++;
  }

  uint32_t odom_id = 0;
  odom_info_.clear();
  for(auto &factor: nfr_solver.rel_pose_factors_) {
    odom_info_.push_back(RelPoseInfo(factor.fi_ptr, factor.fj_ptr, factor.ob, factor.sqrt_info));
    odom_id++;
    if(odom_id < 3) 
      std::cout << "odom info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
  }

  for(auto &factor: nfr_solver.prior_factors_) {
    pivot_kf_ = factor.frame_ptr;
    pivot_sqrt_info_ = factor.sqrt_info;
  }
  
  keyframes_.clear();
  for(auto iter: retain_frames) {
    Frame::Ptr frame = iter.second;
    keyframes_.insert(std::make_pair(frame->id(), frame));
  }
  resort();
  initStruct();

  nfr_time_ += timer.toc();
}

void VectorMap::checkSelf() {
  for(auto &iter: keyframes_) {
    auto parameters = iter.second->Twb().parameters();
    if(parameters.array().isNaN().any() || parameters.array().isInf().any()) {
      printf("Error NAN & INF Values, Keyframe id: %d\n", iter.second->id());
    }
  }

  for(auto &sem_iter: sem_line_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      LineLM::Ptr& lm = lm_iter.second;
      auto parameters = lm->parameters();
      if(parameters.array().isNaN().any() || parameters.array().isInf().any()) {
        printf("Error NAN & INF Values, Keyframe id: %d\n", lm->id());
      }
    }
  }
  for(auto &sem_iter: sem_surf_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      SurfaceLM::Ptr& lm = lm_iter.second;
      auto parameters = lm->parameters();
      if(parameters.array().isNaN().any() || parameters.array().isInf().any()) {
        printf("Error NAN & INF Values, Keyframe id: %d\n", lm->id());
      }
    }
  }

}

void VectorMap::optimizeFullBA(const double weight) {

  std::lock_guard<std::mutex> lock(data_mutex_);

  TicToc timer;
  ceres::Problem problem;
  ceres::LossFunction *huber_loss = new ceres::CauchyLoss(1.0);
  ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();
  ceres::LocalParameterization *line_local_param = new LineLocalParameterization();
  ceres::LocalParameterization *surface_local_param = new SurfaceLocalParameterization();

  std::cout << "all frame size: " << keyframes_.size() << std::endl;

  for(auto &iter: keyframes_) {
    Frame::Ptr frame = iter.second;
    problem.AddParameterBlock(frame->Twb().parameters().data(), 7, pose_local_param);
  }

  Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity() * 1e8;
  PriorPoseFactor *prior_factor = new PriorPoseFactor(pivot_kf_->Twb(), sqrt_info);
  problem.AddResidualBlock(prior_factor, nullptr, pivot_kf_->Twb().parameters().data());

  for(auto &odom_info: odom_info_) {
    Frame::Ptr& fk = odom_info.fi;
    Frame::Ptr& fl = odom_info.fj;
    Transform const Tkl = odom_info.Tij;
    ceres::CostFunction *rel_pose_factor = new RelativePoseFactor(Tkl, odom_info.sqrt_info);
    problem.AddResidualBlock(rel_pose_factor, nullptr, 
      fk->Twb().parameters().data(), fl->Twb().parameters().data());
  }
  
  for(auto &sem_iter: sem_line_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      LineLM::Ptr& lm = lm_iter.second;
      lm->vector2double();
      problem.AddParameterBlock(lm->parameters().data(), 4, line_local_param);
      auto param = lm->getLineInfo().parameters();
      auto obs = lm->getAllObs();
      for(auto &ob_iter: obs) {
        Frame::Ptr& frame = ob_iter.first;
        LineOB::Ptr& ob = ob_iter.second;
        Eigen::Vector3d const point_a = ob->point_a();
        Eigen::Vector3d const point_b = ob->point_b();
        LaserEdge2PFactor *factor = new LaserEdge2PFactor(point_a, point_b, ob->sqrt_info());
        problem.AddResidualBlock(factor, huber_loss, frame->Twb().parameters().data(), lm->parameters().data());
        // std::vector<double*> parameters{frame->Twb().parameters().data(), lm->parameters().data()};
        // factor->CheckJacobian(parameters.data());
      }
    }
  }
  for(auto &sem_iter: sem_surf_lms_) {
    // if(sem_iter.first == ROAD_ID)
    //   continue;
    for(auto &lm_iter: sem_iter.second) {
      SurfaceLM::Ptr& lm = lm_iter.second;
      lm->vector2double();
      auto param = lm->getSurfaceInfo().parameters();
      problem.AddParameterBlock(lm->parameters().data(), 3, surface_local_param);
      auto obs = lm->getAllObs();
      for(auto &ob_iter: obs) {
        Frame::Ptr& frame = ob_iter.first;
        SurfaceOB::Ptr& ob = ob_iter.second;
        std::vector<Eigen::Vector3d> const vertices = ob->vertices();
        LaserSurf3PFactor *factor = new LaserSurf3PFactor(vertices, ob->sqrt_info());
        problem.AddResidualBlock(factor, huber_loss, frame->Twb().parameters().data(), lm->parameters().data());
        // std::vector<double*> parameters{frame->Twb().parameters().data(), lm->parameters().data()};
        // factor->CheckJacobian(parameters.data());
      }
    }
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.trust_region_strategy_type = ceres::DOGLEG;
  options.max_num_iterations = 100;
  options.function_tolerance = 1e-3;
  options.gradient_tolerance = 1e-3;
  options.minimizer_progress_to_stdout = true;
  options.num_threads = 8;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << std::endl;

  for(auto &sem_iter: sem_line_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      lm_iter.second->double2vector();
    }
  }

  for(auto &sem_iter: sem_surf_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      lm_iter.second->double2vector();
    }
  }
  
  ba_time_ += timer.toc();
}

void VectorMap::evaluateError() {
  double line_ape = 0.0, line_me = 0.0;
  int line_num = 0;
  for(auto &sem_iter: sem_line_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      LineLM::Ptr& lm = lm_iter.second;
      auto line_info = lm->getLineInfo();
      auto obs = lm->getAllObs();
      for(auto &ob_iter: obs) {
        Frame::Ptr& frame = ob_iter.first;
        LineOB::Ptr& ob = ob_iter.second;
        Transform Twb = frame->constTwb();
        Eigen::Vector3d const pa = ob->point_a();
        Eigen::Vector3d const pb = ob->point_b();
        Eigen::Vector3d const gpa = Twb * pa;
        Eigen::Vector3d const gpb = Twb * pb;
        double da = (line_info.distance(gpa)).norm();
        double db = (line_info.distance(gpb)).norm();
        double d = std::max(da, db);
        if(d > line_me) {
          line_me = d;
        }
        line_ape += da;
        line_ape += db;
        line_num += 2;
      }
    }
  }
  printf("Line Residual: Max: %lf, Mean: %lf\n", line_me, line_ape / line_num);

  double surf_ape = 0.0, surf_me = 0.0;
  int surf_num = 0;
  for(auto &sem_iter: sem_surf_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      SurfaceLM::Ptr& lm = lm_iter.second;
      auto surf_info = lm->getSurfaceInfo();
      auto obs = lm->getAllObs();
      for(auto &ob_iter: obs) {
        Frame::Ptr& frame = ob_iter.first;
        SurfaceOB::Ptr& ob = ob_iter.second;
        Transform Twb = frame->constTwb();
        auto vertices = ob->vertices();
        Eigen::Vector3d const gpa = Twb * vertices[0];
        Eigen::Vector3d const gpb = Twb * vertices[1];
        Eigen::Vector3d const gpc = Twb * vertices[2];
        double da = std::abs(surf_info.distance(gpa));
        double db = std::abs(surf_info.distance(gpb));
        double dc = std::abs(surf_info.distance(gpc));
        double d = std::max(std::max(da, db), dc);
        if(d > surf_me) {
          surf_me = d;
        }
        surf_ape += da;
        surf_ape += db;
        surf_ape += dc;
        surf_num += 3;
      }
    }
  }
  printf("Surf Residual: Max: %lf, Mean: %lf\n", surf_me, surf_ape / surf_num);

}

void VectorMap::addOdomInfo(const Frame::Ptr &frame, const Eigen::Matrix<double, 6, 6>& sqrt_info) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  if(last_kf_ == nullptr) {
    return;
  }
  Transform Tij = last_kf_->Twb().inverse() * frame->Twb();
  RelPoseInfo info(last_kf_, frame, Tij, sqrt_info);
  // std::cout << "Insert Odom Msg: " << last_kf_->id() << " " << frame->id() << std::endl;
  odom_info_.push_back(info);
}

void VectorMap::addOdomInfo(const RelPoseInfo& info) {
  std::lock_guard<std::mutex> lock(data_mutex_);
  odom_info_.push_back(info);
}

std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> VectorMap::GetMatchInfo() {
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>  match_info;
  // std::cout << "road size: " << sem_surf_lms_[ROAD_ID].size() << std::endl;
  for(auto &lm_vec_iter: sem_line_lms_) {
    for(auto &iter: lm_vec_iter.second) {
      auto lmc = iter.second->centroid();
      auto const &obs = iter.second->getAllObs();
      for(auto &ft: obs) {
        auto const Twb = ft.first->Twb();
        auto fc = Twb * ft.second->centroid();
        match_info.push_back(std::make_pair(lmc, fc));
      }   
    }    
  }

  for(auto &lm_vec_iter: sem_surf_lms_) {
    for(auto &iter: lm_vec_iter.second) {
      auto lmc = iter.second->centroid();
      auto const &obs = iter.second->getAllObs();
      for(auto &ft: obs) {
        auto const Twb = ft.first->Twb();
        auto fc = Twb * ft.second->centroid();
        match_info.push_back(std::make_pair(lmc, fc));
      }   
    }    
  }

  return match_info;
}

void VectorMap::initStruct() {
  pcl::PointXYZL centroid_info;

  sem_line_ct_struct_.clear();
  sem_surf_ct_struct_.clear();

  initQuery();

  for(auto &cls_iter: sem_line_ct_struct_) {
    uint16_t const sem_id = cls_iter.first;
    auto &pair = cls_iter.second;
    auto const& lms = sem_line_lms_[sem_id];
    pair.first.reset(new CentroidCloud());
    if(lms.size() == 0)
      continue;
    pair.first->reserve(lms.size());
    for(auto lm_iter: lms) {
      LineLM::Ptr const& lm = lm_iter.second;
      Eigen::Vector3d const centroid = lm->centroid();
      pcl::PointXYZL centroid_info;
      centroid_info.getVector3fMap() = lm->centroid().cast<float>();
      centroid_info.label = lm->id();
      if(lm->centroid().array().isNaN().any() || lm->centroid().array().isInf().any()) {
        printf("Error NAN & INF Values\n");
      }
      pair.first->push_back(centroid_info);
    }
    pair.second->setInputCloud(pair.first);
  }

  for(auto &cls_iter: sem_surf_ct_struct_) {
    uint16_t const sem_id = cls_iter.first;
    auto &pair = cls_iter.second;
    auto const& lms = sem_surf_lms_[sem_id];
    pair.first.reset(new CentroidCloud());
    if(lms.size() == 0)
      continue;
    pair.first->reserve(lms.size());
    for(auto lm_iter: lms) {
      SurfaceLM::Ptr const& lm = lm_iter.second;
      Eigen::Vector3d const centroid = lm->centroid();
      pcl::PointXYZL centroid_info;
      centroid_info.getVector3fMap() = lm->centroid().cast<float>();
      centroid_info.label = lm->id();
      if(lm->centroid().array().isNaN().any() || lm->centroid().array().isInf().any()) {
        printf("Error NAN & INF Values\n");
      }
      pair.first->push_back(centroid_info);
    }
    pair.second->setInputCloud(pair.first);
  }
}

void VectorMap::insertNewLine(const Frame::Ptr& frame, const LineOB::Ptr& ob) {
  uint32_t id = sem_line_lms_[ob->semantic_type()].size();
  LineLM::Ptr lm = LineLM::Ptr(new LineLM(id, frame, ob));
  ob->SetActive();
  sem_line_lms_[ob->semantic_type()].insert(std::make_pair(lm->id(), lm));
}

void VectorMap::insertNewSurface(const Frame::Ptr& frame, const SurfaceOB::Ptr& ob) {
  uint32_t id = sem_surf_lms_[ob->semantic_type()].size();
  SurfaceLM::Ptr lm = SurfaceLM::Ptr(new SurfaceLM(id, frame, ob));
  ob->SetActive();
  sem_surf_lms_[ob->semantic_type()].insert(std::make_pair(lm->id(), lm));
}

void VectorMap::AssociateLine(const Frame::Ptr& frame) {
  Transform const& Twb = frame->Twb();
  std::vector<LineOB::Ptr> const& line_obs = frame->line_obs();

  for(auto &ob: line_obs) {
    Eigen::Vector3d const centroid = Twb * ob->centroid();
    uint16_t const sem_id = ob->semantic_type();
    LineHashMap &lms = sem_line_lms_[sem_id];
    if(lms.empty()) {
      insertNewLine(frame, ob);
      continue;
    }
    
    std::pair<CentroidCloudPtr, CentroidKdTreePtr> &tree = sem_line_ct_struct_[sem_id];
    if(tree.first == nullptr || tree.second == nullptr)
      continue;
    if(tree.first->empty()) 
      continue;

    pcl::PointXYZL index, query;
    index.getVector3fMap() = centroid.cast<float>();

    std::vector<int> indices;
    std::vector<float> distances;
    tree.second->nearestKSearch(index, 1, indices, distances);
    
    // if(distances[0] > 1.0) {
    //   insertNewLine(frame, ob);
    //   continue;
    // }

    query = tree.first->points[indices[0]];

    uint32_t const lm_id = query.label;
    auto lm_iter = lms.find(lm_id);
    if(lm_iter != lms.end()) {
      LineLM::Ptr lm = lm_iter->second;
      if(lm->associate(ob, Twb)) {
        lm->addNewOb(frame, ob);
        ob->SetActive();
      }      
      else {
        insertNewLine(frame, ob);
      }
    }
  }
}

void VectorMap::AssociateSurface(const Frame::Ptr& frame) {
  Transform const& Twb = frame->Twb();
  std::vector<SurfaceOB::Ptr> const& surface_obs = frame->surface_obs();
  pcl::PointCloud<pcl::PointXYZL> ob_cloud;
  for(uint32_t i = 0; i < surface_obs.size(); ++i) {
    pcl::PointXYZL particle;
    particle.getVector3fMap() = (Twb * surface_obs[i]->centroid()).cast<float>();
    particle.label = i;
    ob_cloud.push_back(particle);
  }
  pcl::search::KdTree<pcl::PointXYZL> ob_tree;
  ob_tree.setInputCloud(ob_cloud.makeShared());

  std::map<uint16_t, std::vector<SurfaceLM::Ptr>> local_map;
  for(auto cls_iter: sem_surf_lms_) {
    uint16_t cls = cls_iter.first;
    if(local_map.find(cls) == local_map.end()) {
      local_map[cls] = std::vector<SurfaceLM::Ptr>();
    }
    for(auto lm_iter: cls_iter.second) {
      SurfaceLM::Ptr lm = lm_iter.second;
      if((lm->centroid() - Twb.p()).norm() < 100.0)
        local_map[cls].push_back(lm);
    }
  }

  for(auto cls_iter: local_map) {
    if(cls_iter.second.size() > 0) {
      std::sort(local_map[cls_iter.first].begin(), local_map[cls_iter.first].end(), [&](const SurfaceLM::Ptr a, const SurfaceLM::Ptr b) {
        return a->getRadius() > b->getRadius();
      });
    }
  }

  std::set<SurfaceOB::Ptr> ob_to_be_merged;
  for(auto cls_iter: local_map) {
    uint16_t cls = cls_iter.first;
    for(auto lm: cls_iter.second) {
      pcl::PointXYZL pivot;
      pivot.getVector3fMap() = lm->centroid().cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      ob_tree.radiusSearch(pivot, lm->getRadius() * std::sqrt(2), indices, distances);

      for(int index = 0; index < indices.size(); ++index) {
        pcl::PointXYZL particle = ob_cloud[indices[index]];
        SurfaceOB::Ptr ob = surface_obs[particle.label];
        if(ob->semantic_type() != lm->semantic_type())
          continue;
        
        if(lm->associate(ob, Twb)) {
          lm->addNewOb(frame, ob);
          ob->SetActive();
          ob_to_be_merged.insert(ob);
        }
      }
    }
  }

  for(auto ob: surface_obs) {
    if(ob_to_be_merged.find(ob) == ob_to_be_merged.end()) {
      insertNewSurface(frame, ob);
    }
  }
}




SurfaceLM::Ptr VectorMap::getLargestSurf(const std::vector<SurfaceLM::Ptr>& surfs) {
  float max_r = -1.0;
  SurfaceLM::Ptr max_surf = nullptr;
  for(auto surf: surfs) {
    if(surf->getRadius() > max_r) {
      max_r = surf->getRadius();
      max_surf = surf;
    }
  }
  return max_surf;
}

std::vector<SurfaceLM::Ptr> VectorMap::extractConsistentSurf(const SurfaceLM::Ptr& ref, const std::vector<SurfaceLM::Ptr>& surfs) {
  Eigen::Vector3d const ref_c = ref->centroid();
  Eigen::Vector3d const ref_n = ref->normal();
  std::vector<SurfaceLM::Ptr> res;
  for(auto surf: surfs) {
    Eigen::Vector3d const cur_c = surf->centroid();
    Eigen::Vector3d const cur_n = surf->normal();
    double ndist = ((Eigen::Matrix3d::Identity() - ref_n * ref_n.transpose()) * (cur_c - ref_c)).norm();
    double theta = std::acos(ref_n.dot(cur_n));
    if(ndist < 0.2 && (theta < M_PI/36 || theta > M_PI-M_PI/36)) {
      res.push_back(surf);
    }
  }
  return res;
}

Json::Value VectorMap::writeFrameObject(const Frame::Ptr frame) {
  Json::Value f;

  const Json::UInt fid = frame->id();
  const Json::UInt64 timestamp = frame->timestamp();
  Transform const Twb = frame->Twb();
  Eigen::Quaterniond Qwb = Twb.q();
  Eigen::Vector3d Pwb = Twb.p();

  f["id"] = Json::Value(fid);
  f["time"] = Json::Value(timestamp);
  f["pose"].append(Pwb.x());f["pose"].append(Pwb.y());f["pose"].append(Pwb.z());
  f["pose"].append(Qwb.x());f["pose"].append(Qwb.y());f["pose"].append(Qwb.z());f["pose"].append(Qwb.w());
  f["file"] = Json::Value(frame->cloud_path_);
  return f;
}

Json::Value VectorMap::writeLineObject(const LineLM::Ptr landmark) {
  Json::Value lm;
  lm["id"] = landmark->id();
  lm["sc"] = kitti_sem_cls_info[landmark->semantic_type()];
  // Eigen::Vector4d const& line_info = landmark->parameters();
  // lm["param"].append(line_info(0));lm["param"].append(line_info(1));lm["param"].append(line_info(2));lm["param"].append(line_info(3)); 
  Eigen::Vector3d const& centroid = landmark->centroid();
  Eigen::Vector3d const& normal = landmark->normal();
  lm["param"].append(centroid(0)); lm["param"].append(centroid(1)); lm["param"].append(centroid(2));
  lm["param"].append(normal(0)); lm["param"].append(normal(1)); lm["param"].append(normal(2));
  auto const& obs = landmark->getAllObs();
  for(auto &ob_iter: obs) {
    uint32_t fid = ob_iter.first->id();
    LineOB::Ptr ob = ob_iter.second;

    Json::Value line;
    line["sc"] = kitti_sem_cls_info[ob->semantic_type()];
    Eigen::Vector3d const& point_a = ob->point_a();
    Eigen::Vector3d const& point_b = ob->point_b();
    Eigen::Matrix4d const& sqrt_info = ob->sqrt_info();
    line["pa"].append(point_a.x()); line["pa"].append(point_a.y()); line["pa"].append(point_a.z());
    line["pb"].append(point_b.x()); line["pb"].append(point_b.y()); line["pb"].append(point_b.z());
    for(int r = 0; r < 4; ++r)
      for(int c = 0; c < 4; ++c)
        line["sqinfo"].append(sqrt_info(r, c));

    // Eigen::Vector3d const& sum = ob->sum_;
    // Eigen::Matrix3d const& squared_sum = ob->squared_sum_;    
    // line["sum"].append(sum(0));
    // line["sum"].append(sum(1));
    // line["sum"].append(sum(2));
    // for(int r = 0; r < 3; ++r)
    //   for(int c = 0; c < 3; ++c)
    //     line["squared_sum"].append(squared_sum(r, c));

    line["num"] = ob->point_num_;

    line["fid"] = fid;
    lm["obs"].append(line);
  }
  return lm;
}

Json::Value VectorMap::writeSurfObject(const SurfaceLM::Ptr landmark) {
  Json::Value lm;
  lm["id"] = landmark->id();
  lm["sc"] = kitti_sem_cls_info[landmark->semantic_type()];
  Eigen::Vector3d const& centroid = landmark->centroid();
  Eigen::Vector3d const& normal = landmark->normal();
  lm["param"].append(centroid(0)); lm["param"].append(centroid(1)); lm["param"].append(centroid(2));
  lm["param"].append(normal(0)); lm["param"].append(normal(1)); lm["param"].append(normal(2));
  lm["radius"] = landmark->getRadius();
  auto const& obs = landmark->getAllObs();
  for(auto &ob_iter: obs) {
    uint32_t fid = ob_iter.first->id();
    SurfaceOB::Ptr ob = ob_iter.second;
    Json::Value surface;
    // Eigen::Vector3d const& centroid = ob->centroid();
    // Eigen::Vector3d const& normal = ob->normal();
    Eigen::Matrix3d const& sqrt_info = ob->sqrt_info();
    surface["sc"] = kitti_sem_cls_info[ob->semantic_type()];
    surface["ra"] = ob->ra();
    surface["rb"] = ob->rb();
    // surface["ct"].append(centroid.x()); surface["ct"].append(centroid.y()); surface["ct"].append(centroid.z());
    // surface["nm"].append(normal.x()); surface["nm"].append(normal.y()); surface["nm"].append(normal.z());

    std::vector<Eigen::Vector3d> vertices = ob->vertices();
    surface["pa"].append(vertices[0].x()); surface["pa"].append(vertices[0].y()); surface["pa"].append(vertices[0].z());
    surface["pb"].append(vertices[1].x()); surface["pb"].append(vertices[1].y()); surface["pb"].append(vertices[1].z());
    surface["pc"].append(vertices[2].x()); surface["pc"].append(vertices[2].y()); surface["pc"].append(vertices[2].z());
    for(int r = 0; r < 3; ++r)
      for(int c = 0; c < 3; ++c)
        surface["sqinfo"].append(sqrt_info(r, c));

    surface["num"] = ob->point_num_;
    surface["fid"] = fid;
    lm["obs"].append(surface);
  }
  return lm;
}

Json::Value VectorMap::writeOdomInfo() {
  Json::Value odinfo;
  for(const RelPoseInfo& info: odom_info_) {
    Json::Value edge;
    edge["ref_id"] = info.fi->id();
    edge["cur_id"] = info.fj->id();
    Eigen::Quaterniond Qwb = info.Tij.q();
    Eigen::Vector3d Pwb = info.Tij.p();
    edge["pose"].append(Pwb.x());edge["pose"].append(Pwb.y());edge["pose"].append(Pwb.z());
    edge["pose"].append(Qwb.x());edge["pose"].append(Qwb.y());edge["pose"].append(Qwb.z());edge["pose"].append(Qwb.w());
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < 6; j++) {
        edge["sqinfo"].append(info.sqrt_info(i, j));
      }
    }
    odinfo["edges"].append(edge);
  }
  return odinfo;
}


Frame::Ptr VectorMap::loadFrame(const Json::Value& f) {
  uint32_t id = f["id"].asUInt();      
  uint64_t timestamp = f["time"].asUInt64();
  Json::Value pose = f["pose"];
  Transform const Twb(Eigen::Vector3d(pose[0].asDouble(), pose[1].asDouble(), pose[2].asDouble()), 
                      Eigen::Quaterniond(pose[6].asDouble(), pose[3].asDouble(), pose[4].asDouble(), pose[5].asDouble()));

  std::string cloud_path = f["file"].asString();
  auto frame = Frame::Ptr(new Frame(id, timestamp, Twb));
  frame->cloud_path_ = cloud_path;
  return frame;
}

LineLM::Ptr VectorMap::loadLineObject(const Json::Value& f, const bool loc) {
  uint32_t id = f["id"].asUInt();      
  uint16_t sem_type = GetSemanticId(f["sc"].asString());
  Json::Value param = f["param"];
  Eigen::Vector3d centroid(param[0].asDouble(), param[1].asDouble(), param[2].asDouble());
  Eigen::Vector3d normal(param[3].asDouble(), param[4].asDouble(), param[5].asDouble());
  LineLM::Ptr lm(new LineLM(id, sem_type, centroid, normal));
  Json::Value obs = f["obs"];

  if(!loc) {
    for(auto line: obs) {
      uint32_t fid = line["fid"].asUInt();
      Json::Value pa = line["pa"], pb = line["pb"];
      uint16_t sem_id = GetSemanticId(line["sc"].asString());
      // Json::Value sgm = line["sgm"];
      Json::Value sqinfo = line["sqinfo"];
      Eigen::Vector3d point_a(pa[0].asDouble(), pa[1].asDouble(), pa[2].asDouble());
      Eigen::Vector3d point_b(pb[0].asDouble(), pb[1].asDouble(), pb[2].asDouble());
      Eigen::Matrix4d sqrt_info;
      for(int i = 0; i < 4; ++i) {
        for(int j = 0; j < 4; ++j) {
          sqrt_info(i, j) = sqinfo[i*4+j].asDouble();
        }
      }

      uint32_t point_num = line["num"].asUInt();
      LineOB::Ptr line_ob = LineOB::Ptr(new LineOB(sem_id, point_a, point_b, point_num));
      line_ob->setSqrtInfo(sqrt_info);
      lm->addNewOb(keyframes_[fid], line_ob);
    }
    lm->solveByObs();
  }
  return lm;
}


SurfaceLM::Ptr VectorMap::loadSurfObject(const Json::Value& f, const bool loc) {
  uint32_t id = f["id"].asUInt();      
  uint16_t sem_type = GetSemanticId(f["sc"].asString());
  Json::Value param = f["param"];
  double radius = f["radius"].asDouble();
  Eigen::Vector3d centroid(param[0].asDouble(), param[1].asDouble(), param[2].asDouble());
  Eigen::Vector3d normal(param[3].asDouble(), param[4].asDouble(), param[5].asDouble());
  SurfaceLM::Ptr lm(new SurfaceLM(id, sem_type, radius, centroid, normal));
  Json::Value obs = f["obs"];

  if(!loc) {
    for(auto surf: obs) {
      uint32_t ob_id = surf["id"].asUInt();
      uint32_t fid = surf["fid"].asUInt();
      uint16_t sem_id = GetSemanticId(surf["sc"].asString());

      Json::Value sqinfo = surf["sqinfo"];
      Json::Value pa = surf["pa"], pb = surf["pb"], pc = surf["pc"];
      Eigen::Vector3d point_a(pa[0].asDouble(), pa[1].asDouble(), pa[2].asDouble());
      Eigen::Vector3d point_b(pb[0].asDouble(), pb[1].asDouble(), pb[2].asDouble());
      Eigen::Vector3d point_c(pc[0].asDouble(), pc[1].asDouble(), pc[2].asDouble());
      std::vector<Eigen::Vector3d> vertices{point_a, point_b, point_c};

      Eigen::Matrix3d sqrt_info;
      for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
          sqrt_info(i, j) = sqinfo[i*3+j].asDouble();
        }
      }

      float ra = surf["ra"].asFloat(), rb = surf["rb"].asFloat();
      Json::Value s1 = surf["sum"], s2 = surf["squared_sum"];
      Eigen::Vector3d sum(s1[0].asDouble(), s1[1].asDouble(),s1[2].asDouble());
      Eigen::Matrix3d squared_sum;
      for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 3; ++j) {
          squared_sum(i, j) = s2[i*3+j].asDouble();
        }
      }
      uint32_t point_num = surf["num"].asUInt();
      SurfaceOB::Ptr surf_ob = SurfaceOB::Ptr(new SurfaceOB(sem_id, vertices, point_num));
      surf_ob->ra_ = ra;
      surf_ob->rb_ = rb;
      surf_ob->setSqrtInfo(sqrt_info);
      lm->addNewOb(keyframes_[fid], surf_ob);
    }
    lm->solveByObs();
  }
  return lm;
}


void VectorMap::loadOdomInfo(const Json::Value &root, const bool loc) {
  
  Json::Value edges = root["odom_info"]["edges"];
  if(keyframes_.size() == 0) {
    return;
  }

  for(auto edge: edges) {
    uint32_t ref_id = edge["ref_id"].asUInt();
    uint32_t cur_id = edge["cur_id"].asUInt();

    Json::Value pose = edge["pose"];
    Transform const Tij(Eigen::Vector3d(pose[0].asDouble(), pose[1].asDouble(), pose[2].asDouble()), 
                        Eigen::Quaterniond(pose[6].asDouble(), pose[3].asDouble(), pose[4].asDouble(), pose[5].asDouble()));

    Frame::Ptr fi = keyframes_[ref_id];
    Frame::Ptr fj = keyframes_[cur_id];

    Eigen::Matrix<double, 6, 6> sqrt_info;
    Json::Value sqinfo = edge["sqinfo"];
    for(int i = 0; i < 6; i++) {
      for(int j = 0; j < 6; j++) {
        sqrt_info(i, j) = sqinfo[i*6+j].asDouble();
      }
    }
    RelPoseInfo info(fi, fj, Tij, sqrt_info);
    odom_info_.push_back(info);
  }
}


void VectorMap::saveTumTrajectory(const std::string& filename)
{
  std::map<uint64_t, Transform> trajectory;
  for(auto &f_iter: keyframes_) {
    uint64_t time = f_iter.second->timestamp();
    if(trajectory.find(time) == trajectory.end()) {
      trajectory.insert(std::make_pair(time, f_iter.second->Twb()));
    }
  }

  std::ofstream f(filename, std::ios::out);
  f.setf(std::ios::fixed, std::ios::floatfield);
  f.precision(16);
  for(auto &f_iter: trajectory) {
    // double timestamp = (double)f_iter.first / 1e6;
    // double timestamp = (double)f_iter.first;
    uint64_t timestamp = f_iter.first;
    // f.precision(6);
    // f << timestamp << " ";
    // f.precision(10);
    // uint64_t timestamp = f_iter.first;
    Eigen::Quaterniond qwb = f_iter.second.q();
    Eigen::Vector3d twb = f_iter.second.p();
    f << timestamp << " " << twb.x() << " " << twb.y() << " " << twb.z() << " " << qwb.x() << " " << qwb.y() << " " << qwb.z() << " " << qwb.w() << std::endl;
  }
  f.close();  
}

void VectorMap::saveTrajAndCloudPath(const std::string& filename) {
  std::map<uint64_t, std::pair<Transform, std::string>> trajectory;
  for(auto &f_iter: keyframes_) {
    uint64_t time = f_iter.second->timestamp();
    if(trajectory.find(time) == trajectory.end()) {
      trajectory.insert(std::make_pair(time, std::make_pair(f_iter.second->Twb(), f_iter.second->cloud_path_)));
    }
  }

  std::ofstream f(filename, std::ios::out);
  f.setf(std::ios::fixed, std::ios::floatfield);
  f.precision(16);
  for(auto &f_iter: trajectory) {
    uint64_t timestamp = f_iter.first;
    Eigen::Quaterniond qwb = f_iter.second.first.q();
    Eigen::Vector3d twb = f_iter.second.first.p();
    std::string cloud_path = f_iter.second.second;
    std::cout << cloud_path << std::endl;
    f << timestamp << " " << twb.x() << " " << twb.y() << " " << twb.z() << " " << qwb.x() << " " << qwb.y() << " " << qwb.z() << " " << qwb.w() << " " << cloud_path << std::endl;
  }
  f.close();  
}

void VectorMap::saveSyncBenchmark(const std::string& filename, const std::map<uint64_t, Transform>& gt) {

  std::map<uint64_t, Transform> trajectory;
  for(auto &f_iter: keyframes_) {
    uint64_t time = f_iter.second->timestamp();
    if(trajectory.find(time) == trajectory.end()) {
      trajectory.insert(std::make_pair(time, f_iter.second->Twb()));
    }
  }

  std::map<uint64_t, Transform> sync_trajectory;
  for(auto &f_iter: trajectory) {
    uint64_t time = f_iter.first;
    std::cout << time << std::endl;
    auto gt_iter = gt.find(time);
    if(gt_iter != gt.end()) {
      const Transform Twb = gt_iter->second;
      sync_trajectory.insert(std::make_pair(time, Twb));      
    }
  }

  std::ofstream f(filename, std::ios::out);
  for(auto &f_iter: sync_trajectory) {
    double timestamp = (double)f_iter.first / 1e6;
    Eigen::Quaterniond qwb = f_iter.second.q();
    Eigen::Vector3d twb = f_iter.second.p();
    f << std::setprecision(20) << timestamp << std::setprecision(10) << " " << twb.x() << " " << twb.y() << " " << twb.z()
          << " " << qwb.x() << " " << qwb.y() << " " << qwb.z() << " " << qwb.w() << std::endl;
  }
  f.close();

}

void VectorMap::saveSummary(const std::string& summary_file) {
  uint32_t kf_size = keyframes_.size();
  uint32_t pole_lm_size = sem_line_lms_[POLE_ID].size();
  uint32_t road_lm_size = sem_surf_lms_[ROAD_ID].size();
  uint32_t building_lm_size = sem_surf_lms_[BUILDING_ID].size();

  #define KF_STORAGE_SIZE 40 // pose(7*4) + id(4) + timestamp(8)
  #define LINE_LM_STORAGE_SIZE 30 // id(4) + class(2) + centroid(3*4) + normal(3*4)
  #define SURF_LM_STORAGE_SIZE 34 // id(4) + class(2) + centroid(3*4) + normal(3*4) + radius(4)
  #define LINE_OB_STORAGE_SIZE 72 // pa(12) + pb(12) + num(4) + sqinfo(10*4, sym mat) + lmid(4)
  #define SURF_OB_STORAGE_SIZE 76 // pa(12) + pb(12) + pc(12) + num(4) + sqinfo(6*4, sym mat) + lmid(4) + ra(4) + rb(4)
  #define ODOM_OB_STORAGE_SIZE 120 // rel_pose(7*4) + fi(4) + fj(4) + sqinfo(21*4, sym mat)

  uint32_t kf_info_storage = kf_size * KF_STORAGE_SIZE;
  uint32_t odom_info_storage = odom_info_.size() * ODOM_OB_STORAGE_SIZE;
  uint32_t line_lm_storage = pole_lm_size * LINE_LM_STORAGE_SIZE;
  uint32_t surf_lm_storage = (road_lm_size + building_lm_size) * SURF_LM_STORAGE_SIZE;
  uint32_t line_ob_storage = 0, surf_ob_storage = 0;

  for(auto &sem_iter: sem_line_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      LineLM::Ptr& lm = lm_iter.second;
      line_ob_storage += lm->getAllObs().size() * LINE_OB_STORAGE_SIZE;
    }
  }
  for(auto &sem_iter: sem_surf_lms_) {
    for(auto &lm_iter: sem_iter.second) {
      SurfaceLM::Ptr& lm = lm_iter.second;
      surf_ob_storage += lm->getAllObs().size() * SURF_OB_STORAGE_SIZE;
    }
  }

  uint32_t total_storage = kf_info_storage + odom_info_storage + line_lm_storage + surf_lm_storage + line_ob_storage + surf_ob_storage;

  Json::Value root;
  root["name"] = Json::Value("SLIM");
  root["kf_size"] = Json::Value(kf_size);
  root["pole_lm_size"] = Json::Value(pole_lm_size);
  root["road_lm_size"] = Json::Value(road_lm_size);
  root["building_lm_size"] = Json::Value(building_lm_size);

  root["kf_info_storage"] = Json::Value(kf_info_storage);
  root["odom_info_storage"] = Json::Value(odom_info_storage);
  root["line_lm_storage"] = Json::Value(line_lm_storage);
  root["surf_lm_storage"] = Json::Value(surf_lm_storage);
  root["line_ob_storage"] = Json::Value(line_ob_storage);
  root["surf_ob_storage"] = Json::Value(surf_ob_storage);
  root["total_storage"] = Json::Value(total_storage);
  root["ba_time"] = Json::Value(ba_time_);
  root["nfr_time"] = Json::Value(nfr_time_);


  std::ofstream os;
  os.open(summary_file, std::ios::out);
  if(!os.is_open()) {
    std::runtime_error("Error: Save Summary Failed...");
    return;
  }
  Json::StyledWriter sw;
  os << sw.write(root);
  os.close();

  // printf("***************************** Map Info ****************************\n");
  // printf("File Name: %s\n", name.c_str());
  // printf("Trajectory Length: %lf\n", length);
  // printf("Frame Size: %d, Line Landmark Size: %d, Surface Landmark Size %d\n", frame_size, line_lm_size, surf_lm_size);
  // for(auto cls_iter: line_lm_size_map) {
  //   printf("Class: %s, Size: %d\n", GetSemanticName(cls_iter.first).c_str(), cls_iter.second);
  // }
  // for(auto cls_iter: surf_lm_size_map) {
  //   printf("Class: %s, Size: %d\n", GetSemanticName(cls_iter.first).c_str(), cls_iter.second);
  // }
  // printf("Frame Line Observation: %d, Surface Observation: %d\n", line_ob_size, surf_ob_size);
  // double storage = 0.0;
  // storage = (frame_size * 40) + (line_lm_size * 26) + (surf_lm_size * 38) + (line_lm_size * 30) + (surf_lm_size * 30);
  // double loc_storage = (line_lm_size * 30) + (surf_lm_size * 30);
  // printf("Map Storage: %lf MB, Map For Localizaiton Storage: %lf MB\n", storage/1e6, loc_storage/1e6);
  // printf("Map Storage Ratio: %lf KB/km, Map For Localizaiton Storage Ratio: %lf KB/km\n", (storage) / (length), (loc_storage) / (length));
  // printf("First Timestamp: %lf, Last Timestamp: %lf\n", (double)odom_info.front().fi->timestamp()/1e6, (double)odom_info.back().fj->timestamp()/1e6);
}

void VectorMap::printInfo() {
  printf("[VectorMap Info]\n");
  for(auto cls_iter: sem_line_lms_) {
    std::cout << "Line Feature: " << GetSemanticName(cls_iter.first) << " Size: " << cls_iter.second.size() << std::endl;
  }
  for(auto cls_iter: sem_surf_lms_) {
    std::cout << "Surf Feature: " << GetSemanticName(cls_iter.first) << " Size: " << cls_iter.second.size() << std::endl;
  }
}


} // namespace SLIM