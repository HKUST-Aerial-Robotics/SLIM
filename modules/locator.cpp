#include "locator.h"

namespace SLIM {

void Locator::setMap(const VectorMap::Ptr& map) {
  map_ = map;
  map_->divideBlocks(20.0f, 40.0f);
  blocks_ = map_->getBlocks();
  for(auto &block: blocks_) {
    SceneGraph::Ptr graph(new SceneGraph);
    graph->buildFromBlock(block);
    graphs_.push_back(graph);
  }
}

void Locator::setViewer(const Viewer::Ptr& viewer) {
  viewer_ = viewer;
}

void Locator::setTrajFileName(const std::string& traj_file_name) {
  traj_file_name_ = traj_file_name;
}

bool Locator::solveRelativePose(const SceneGraph::Ptr& gref, const SceneGraph::Ptr& gcur, 
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
  if(std::abs(euler(1)) > 20.0 / 360.0 * M_PI || std::abs(euler(2)) > 20.0 / 360.0 * M_PI) {
    printf("[MapMerging][SolveRelativePose] Failed! Too large roll or pitch. Time Cost: %lf ms.\n", timer.toc());
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

bool Locator::refineRelativePose(const Block::Ptr& block, 
                                 const pcl::PointCloud<pcl::PointXYZI>::Ptr& sem_cloud, 
                                 Transform& Tij) {
  // refine pose (using knn search and semi-dense centroid cloud)
  // for every landmark in block j, find the nearest landmark (using the coarse Tij) in current map
  TicToc timer;
  auto &sem_line_lms = block->getSemLines();
  auto &sem_surf_lms = block->getSemSurfaces();
  auto &sem_line_struct = block->getSemLineStruct();
  auto &sem_surf_struct = block->getSemSurfStruct();

  double line_thres = 10.0, surface_thres = 10.0;
  int iter = 0;

  bool converge = false;
  double line_inlier_ratio = 0.0, surf_inlier_ratio = 0.0;
  for(; iter < 5; ++iter) {
    ceres::Problem problem;
    ceres::LossFunction *huber_loss = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  
    problem.AddParameterBlock(Tij.parameters().data(), 7, pose_local_param);

    uint32_t line_lm_cnt = 0, surf_lm_cnt = 0, line_lm_valid_cnt = 0, surf_lm_valid_cnt = 0;
    for(auto &point: sem_cloud->points) {
      uint16_t sem_id = std::round(point.intensity);
      if(sem_id == POLE_ID) {
        CentroidCloudPtr &nodes = sem_line_struct[sem_id].first;
        if(nodes.get() == nullptr) continue;
        if(nodes->empty()) continue;
        CentroidKdTreePtr &tree = sem_line_struct[sem_id].second;
        std::unordered_map<uint32_t, Block::Node::Ptr> &lms = sem_line_lms[sem_id];
        Eigen::Vector3d const cj = point.getVector3fMap().cast<double>();
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
        if(distances[0] > 10 * 10)
          continue;

        uint32_t const lm_id = nodes->points[indices[0]].label;
        Block::Node::Ptr lm = lms[lm_id];
        Eigen::Vector3d const ci = lm->centroid;
        Eigen::Vector3d const ni = lm->normal;
        Eigen::Matrix3d const nmat = Eigen::Matrix3d::Identity() - ni * ni.transpose();
        Eigen::Vector3d const r = nmat * (qj - ci);
        if(r.norm() < line_thres) {
          LaserPointToLineFactor *f = new LaserPointToLineFactor(cj, ci, ni, 1/0.2);
          problem.AddResidualBlock(f, huber_loss, Tij.parameters().data());
          line_lm_valid_cnt++;
        }
      }
      else {
        CentroidCloudPtr &nodes = sem_surf_struct[sem_id].first;
        if(nodes.get() == nullptr) continue;
        if(nodes->empty()) continue;
        CentroidKdTreePtr &tree = sem_surf_struct[sem_id].second;
        std::unordered_map<uint32_t, Block::Node::Ptr> &lms = sem_surf_lms[sem_id];
        Eigen::Vector3d const cj = point.getVector3fMap().cast<double>();
        Eigen::Vector3d const qj = Tij * cj;
        surf_lm_cnt++;

        // nn search
        pcl::PointXYZL index;
        index.getVector3fMap() = cj.cast<float>();
        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(index, 1, indices, distances);
        if(indices.empty()) 
          continue;
        if(distances[0] > 10 * 10)
          continue;

        uint32_t const lm_id = nodes->points[indices[0]].label;
        Block::Node::Ptr lm = lms[lm_id];
        Eigen::Vector3d const ci = lm->centroid;
        Eigen::Vector3d const ni = lm->normal;
        SurfaceInfo surface(ci, ni);
        double const ndist = surface.distance(qj);
        Eigen::Vector3d const pedal = surface.pedal(qj);
        double const pdist = (ci - pedal).norm();
        // double const theta = std::acos(ni.dot(Tij.dcm() * nj));
        std::cout << "ndist: " << ndist << std::endl;
        if(pdist < GetSemanticResolution(sem_id) && std::abs(ndist) < surface_thres) { // && ndist < 5.0) {
          LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(cj, ci, ni, 1/0.2);
          problem.AddResidualBlock(f, huber_loss, Tij.parameters().data());
          surf_lm_valid_cnt++;
        }
      }
    }
    line_thres = std::max(0.3, line_thres/4.0);
    surface_thres = std::max(0.3, surface_thres/4.0);

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
    if(std::abs(euler(1)) > 20.0 / 360.0 * M_PI || std::abs(euler(2)) > 20.0 / 360.0 * M_PI) {
      // printf("[MapMerging][SolveRelativePose] Failed! Too large roll or pitch. Time Cost: %lf ms.\n", timer.toc());
      return false;
    }
  }
  cur_line_ratio_ = line_inlier_ratio;
  cur_surf_ratio_ = surf_inlier_ratio;

  if(line_inlier_ratio < 0.3 || surf_inlier_ratio < 0.3) {
    printf("[MapMerging][RefineRelativePose] Failed! Too little landmark registration inlier ratio! (%lf, %lf)...\n", line_inlier_ratio, surf_inlier_ratio);
    return false;
  }
  // std::cout << "[MapMerging][RefineRelativePose] Successful! Refine Pose: " << Tij.p().transpose() << " " << R2ypr(Tij.dcm()).transpose() << std::endl;
  printf("[MapMerging][RefineRelativePose] Successful! Line/Surface Ratio: %lf%/%lf%\n", line_inlier_ratio * 100, surf_inlier_ratio * 100);
  // printf("[MapMerging][RefineRelativePose] Successful! Time Cost: %lf ms. Iteration: %d.\n", timer.toc(), iter);
  return true;
}

bool Locator::solve(uint64_t timestamp, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  TicToc timer;
  Transform Twb = cur_Twb_ * delta_Twb_;
  // cv::Mat bottom(100, 100, CV_8U, cv::Scalar(255));

  auto &sem_line_lms = map_->getSemLines();
  auto &sem_surf_lms = map_->getSemSurfaces();
  auto &sem_line_struct = map_->getSemLineStruct();
  auto &sem_surf_struct = map_->getSemSurfStruct();
  double line_thres = 5, surface_thres = 5;
  int iter = 0;

  // std::cout << "[solve]: " << std::endl << Twb.matrix() << std::endl;
  // std::cout << "cloud size: " << cloud->size() << std::endl;
  
  bool converge = false;
  double line_inlier_ratio = 0.0, surf_inlier_ratio = 0.0;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> loc_residuals;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr effective_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  for(; iter < 3; ++iter) {
    ceres::Problem problem;
    ceres::LossFunction *huber_loss = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  
    problem.AddParameterBlock(Twb.parameters().data(), 7, pose_local_param);

    effective_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    loc_residuals.clear();
    uint32_t line_lm_cnt = 0, surf_lm_cnt = 0, line_lm_valid_cnt = 0, surf_lm_valid_cnt = 0;
    for(auto &point: cloud->points) {
      uint16_t sem_id = std::round(point.intensity);
      if(sem_id == POLE_ID) {
        CentroidCloudPtr &nodes = sem_line_struct[sem_id].first;
        if(nodes.get() == nullptr) continue;
        if(nodes->empty()) continue;
        CentroidKdTreePtr &tree = sem_line_struct[sem_id].second;
        LineHashMap &lms = sem_line_lms[sem_id];
        Eigen::Vector3d const cj = point.getVector3fMap().cast<double>();
        Eigen::Vector3d const qj = Twb * cj;
        line_lm_cnt++;
                
        // nn search
        pcl::PointXYZL index;
        index.getVector3fMap() = qj.cast<float>();
        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(index, 1, indices, distances);
        if(indices.empty()) 
          continue;
        if(distances[0] > 10 * 10)
          continue;

        uint32_t const lm_id = nodes->points[indices[0]].label;
        LineLM::Ptr lm = lms[lm_id];
        Eigen::Vector3d const ci = lm->centroid();
        Eigen::Vector3d const ni = lm->normal();
        Eigen::Matrix3d const nmat = Eigen::Matrix3d::Identity() - ni * ni.transpose();
        Eigen::Vector3d const r = nmat * (qj - ci);
        LineInfo line_info = lm->getLineInfo();
        Eigen::Vector3d const pedal = line_info.pedal(qj);
        if(r.norm() < line_thres) {
          LaserPointToLineFactor *f = new LaserPointToLineFactor(cj, ci, ni, 1/0.2);
          problem.AddResidualBlock(f, huber_loss, Twb.parameters().data());
          loc_residuals.push_back(std::make_pair(qj, pedal));
          line_lm_valid_cnt++;

          pcl::PointXYZRGB particle;
          particle.getVector3fMap() = cj.cast<float>();
          particle.r = CV_COLOR_INDIGO[0];
          particle.g = CV_COLOR_INDIGO[1];
          particle.b = CV_COLOR_INDIGO[2];
          effective_cloud->push_back(particle);
        }
      }
      else {
        CentroidCloudPtr &nodes = sem_surf_struct[sem_id].first;
        if(nodes.get() == nullptr) continue;
        if(nodes->empty()) continue;
        CentroidKdTreePtr &tree = sem_surf_struct[sem_id].second;
        SurfaceHashMap &lms = sem_surf_lms[sem_id];
        Eigen::Vector3d const cj = point.getVector3fMap().cast<double>();
        Eigen::Vector3d const qj = Twb * cj;
        surf_lm_cnt++;

        // nn search
        pcl::PointXYZL index;
        index.getVector3fMap() = qj.cast<float>();
        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(index, 1, indices, distances);
        if(indices.empty()) 
          continue;
        if(distances[0] > 5 * 5)
          continue;

        uint32_t const lm_id = nodes->points[indices[0]].label;
        SurfaceLM::Ptr lm = lms[lm_id];
        Eigen::Vector3d const ci = lm->centroid();
        Eigen::Vector3d const ni = lm->normal();
        SurfaceInfo surface(ci, ni);
        double const ndist = surface.distance(qj);
        Eigen::Vector3d const pedal = surface.pedal(qj);
        double const pdist = (ci - pedal).norm();
        if(pdist < GetSemanticResolution(sem_id) && std::abs(ndist) < surface_thres) { // && ndist < 5.0) {
          LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(cj, ci, ni, 1/0.2);
          problem.AddResidualBlock(f, huber_loss, Twb.parameters().data());
          loc_residuals.push_back(std::make_pair(qj, pedal));
          surf_lm_valid_cnt++;

          pcl::PointXYZRGB particle;
          particle.getVector3fMap() = cj.cast<float>();
          if(sem_id == BUILDING_ID || sem_id == FENCE_ID) {
            particle.r = CV_COLOR_ORANGERED[0];
            particle.g = CV_COLOR_ORANGERED[1];
            particle.b = CV_COLOR_ORANGERED[2];
          }
          else {
            particle.r = CV_COLOR_PURPLE[0];
            particle.g = CV_COLOR_PURPLE[1];
            particle.b = CV_COLOR_PURPLE[2];
          }

          effective_cloud->push_back(particle);
        }
      }
    }
    line_thres = std::max(0.3, line_thres/3.0);
    surface_thres = std::max(0.3, surface_thres/3.0);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.function_tolerance = 1e-4;
    options.gradient_tolerance = 1e-4;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    line_inlier_ratio = (line_lm_cnt == 0) ? 1 : (double)line_lm_valid_cnt/line_lm_cnt;
    surf_inlier_ratio = (surf_lm_cnt == 0) ? 0 : (double)surf_lm_valid_cnt/surf_lm_cnt;
  }
  cur_line_ratio_ = line_inlier_ratio;
  cur_surf_ratio_ = surf_inlier_ratio;
  // viewer_->SetResiduals(loc_residuals);
  cur_Twb_ = Twb;
  trajectory_.insert(std::make_pair(timestamp, cur_Twb_));
  delta_Twb_ = last_Twb_.inverse() * cur_Twb_;
  last_Twb_ = cur_Twb_;  

  initialized_ = true;
  Eigen::Matrix4f affine = cur_Twb_.matrix().cast<float>();
  for(int j = 0; j < effective_cloud->size(); ++j) {
    effective_cloud->points[j].getVector4fMap() = affine * effective_cloud->points[j].getVector4fMap();
  }

  std::cout << "[Locator][RefineRelativePose]" << std::endl << Twb.matrix() << std::endl;
  printf("[MapMerging][RefineRelativePose] Successful! Line/Surface Ratio: %lf%/%lf%\n", line_inlier_ratio * 100, surf_inlier_ratio * 100);
  printf("[MapMerging][RefineRelativePose] Successful! Time Cost: %lf ms. Iteration: %d.\n", timer.toc(), iter);
  return true;

}


bool Locator::solve(uint64_t timestamp, 
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& pole, 
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& road,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& building
) {
  TicToc timer;
  Transform Twb = cur_Twb_ * delta_Twb_;

  auto &sem_line_lms = map_->getSemLines();
  auto &sem_surf_lms = map_->getSemSurfaces();
  auto &sem_line_struct = map_->getSemLineStruct();
  auto &sem_surf_struct = map_->getSemSurfStruct();
  double line_thres = 5, surface_thres = 5;
  int iter = 0;

  bool converge = false;
  double line_inlier_ratio = 0.0, surf_inlier_ratio = 0.0;
  std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> loc_residuals;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr effective_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

  for(; iter < 3; ++iter) {
    ceres::Problem problem;
    ceres::LossFunction *huber_loss = new ceres::CauchyLoss(1.0);
    ceres::LocalParameterization *pose_local_param = new PoseLocalParameterization();  
    problem.AddParameterBlock(Twb.parameters().data(), 7, pose_local_param);

    effective_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>());

    loc_residuals.clear();
    uint32_t line_lm_cnt = 0, surf_lm_cnt = 0, line_lm_valid_cnt = 0, surf_lm_valid_cnt = 0;

    for(auto &point: pole->points) {
      CentroidCloudPtr &nodes = sem_line_struct[POLE_ID].first;
      if(nodes.get() == nullptr) continue;
      if(nodes->empty()) continue;
      CentroidKdTreePtr &tree = sem_line_struct[POLE_ID].second;
      LineHashMap &lms = sem_line_lms[POLE_ID];
      Eigen::Vector3d const cj = point.getVector3fMap().cast<double>();
      Eigen::Vector3d const qj = Twb * cj;
      line_lm_cnt++;

      // nn search
      pcl::PointXYZL index;
      index.getVector3fMap() = qj.cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      tree->nearestKSearch(index, 1, indices, distances);
      if(indices.empty()) 
        continue;
      if(distances[0] > 10 * 10)
        continue;

      uint32_t const lm_id = nodes->points[indices[0]].label;
      LineLM::Ptr lm = lms[lm_id];
      Eigen::Vector3d const ci = lm->centroid();
      Eigen::Vector3d const ni = lm->normal();
      Eigen::Matrix3d const nmat = Eigen::Matrix3d::Identity() - ni * ni.transpose();
      Eigen::Vector3d const r = nmat * (qj - ci);
      LineInfo line_info = lm->getLineInfo();
      Eigen::Vector3d const pedal = line_info.pedal(qj);
      if(r.norm() < line_thres) {
        LaserPointToLineFactor *f = new LaserPointToLineFactor(cj, ci, ni, 1/0.2);
        problem.AddResidualBlock(f, huber_loss, Twb.parameters().data());
        loc_residuals.push_back(std::make_pair(qj, pedal));
        line_lm_valid_cnt++;

        pcl::PointXYZRGB particle;
        particle.getVector3fMap() = cj.cast<float>();
        particle.r = CV_COLOR_GREEN[0];
        particle.g = CV_COLOR_GREEN[1];
        particle.b = CV_COLOR_GREEN[2];
        effective_cloud->push_back(particle);
      }
    }


    for(auto &point: road->points) {
      CentroidCloudPtr &nodes = sem_surf_struct[ROAD_ID].first;
      if(nodes.get() == nullptr) continue;
      if(nodes->empty()) continue;
      CentroidKdTreePtr &tree = sem_surf_struct[ROAD_ID].second;
      SurfaceHashMap &lms = sem_surf_lms[ROAD_ID];
      Eigen::Vector3d const cj = point.getVector3fMap().cast<double>();
      Eigen::Vector3d const qj = Twb * cj;
      surf_lm_cnt++;

      // nn search
      pcl::PointXYZL index;
      index.getVector3fMap() = qj.cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      tree->nearestKSearch(index, 1, indices, distances);
      if(indices.empty()) 
        continue;
      if(distances[0] > 5 * 5)
        continue;

      uint32_t const lm_id = nodes->points[indices[0]].label;
      SurfaceLM::Ptr lm = lms[lm_id];
      Eigen::Vector3d const ci = lm->centroid();
      Eigen::Vector3d const ni = lm->normal();
      SurfaceInfo surface(ci, ni);
      double const ndist = surface.distance(qj);
      Eigen::Vector3d const pedal = surface.pedal(qj);
      double const pdist = (ci - pedal).norm();
      if(pdist < lm->getRadius() && std::abs(ndist) < surface_thres) {
        LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(cj, ci, ni, 1/0.2);
        problem.AddResidualBlock(f, huber_loss, Twb.parameters().data());
        loc_residuals.push_back(std::make_pair(qj, pedal));
        surf_lm_valid_cnt++;

        pcl::PointXYZRGB particle;
        particle.getVector3fMap() = cj.cast<float>();
        particle.r = CV_COLOR_LIGHTGREEN[0];
        particle.g = CV_COLOR_LIGHTGREEN[1];
        particle.b = CV_COLOR_LIGHTGREEN[2];
        effective_cloud->push_back(particle);
      }
    }

    for(auto &point: building->points) {
      CentroidCloudPtr &nodes = sem_surf_struct[BUILDING_ID].first;
      if(nodes.get() == nullptr) continue;
      if(nodes->empty()) continue;
      CentroidKdTreePtr &tree = sem_surf_struct[BUILDING_ID].second;
      SurfaceHashMap &lms = sem_surf_lms[BUILDING_ID];
      Eigen::Vector3d const cj = point.getVector3fMap().cast<double>();
      Eigen::Vector3d const qj = Twb * cj;
      surf_lm_cnt++;

      // nn search
      pcl::PointXYZL index;
      index.getVector3fMap() = qj.cast<float>();
      std::vector<int> indices;
      std::vector<float> distances;
      tree->nearestKSearch(index, 1, indices, distances);
      if(indices.empty()) 
        continue;
      if(distances[0] > 5 * 5)
        continue;

      uint32_t const lm_id = nodes->points[indices[0]].label;
      SurfaceLM::Ptr lm = lms[lm_id];
      Eigen::Vector3d const ci = lm->centroid();
      Eigen::Vector3d const ni = lm->normal();
      SurfaceInfo surface(ci, ni);
      double const ndist = surface.distance(qj);
      Eigen::Vector3d const pedal = surface.pedal(qj);
      double const pdist = (ci - pedal).norm();
      if(pdist < lm->getRadius() && std::abs(ndist) < surface_thres) {
        LaserPointToSurfaceFactor *f = new LaserPointToSurfaceFactor(cj, ci, ni, 1/0.2);
        problem.AddResidualBlock(f, huber_loss, Twb.parameters().data());
        loc_residuals.push_back(std::make_pair(qj, pedal));
        surf_lm_valid_cnt++;

        pcl::PointXYZRGB particle;
        particle.getVector3fMap() = cj.cast<float>();
        particle.r = CV_COLOR_LIGHTGREEN[0];
        particle.g = CV_COLOR_LIGHTGREEN[1];
        particle.b = CV_COLOR_LIGHTGREEN[2];
        effective_cloud->push_back(particle);
      }
    }


    line_thres = std::max(0.3, line_thres/4.0);
    surface_thres = std::max(0.3, surface_thres/4.0);

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
  }
  cur_line_ratio_ = line_inlier_ratio;
  cur_surf_ratio_ = surf_inlier_ratio;
  cur_Twb_ = Twb;
  trajectory_.insert(std::make_pair(timestamp, cur_Twb_));
  delta_Twb_ = last_Twb_.inverse() * cur_Twb_;
  last_Twb_ = cur_Twb_;  

  initialized_ = true;

  Eigen::Matrix4f affine = cur_Twb_.matrix().cast<float>();
  for(int j = 0; j < effective_cloud->size(); ++j) {
    effective_cloud->points[j].getVector4fMap() = affine * effective_cloud->points[j].getVector4fMap();
  }

  // viewer_->RemoveAllCloud();
  viewer_->SetCurrentPose(cur_Twb_);
  printf("Localization Time Cost: %lf\n", timer.toc());
  return true;
}

void Locator::saveTumTrajectory(const std::string& filename)
{
  std::ofstream f(filename, std::ios::out);
  f.setf(std::ios::fixed, std::ios::floatfield);
  f.precision(16);
  for(auto &f_iter: trajectory_) {
    uint64_t timestamp = f_iter.first;
    Eigen::Quaterniond qwb = f_iter.second.q();
    Eigen::Vector3d twb = f_iter.second.p();
    f << timestamp << " " << twb.x() << " " << twb.y() << " " << twb.z() << " " << qwb.x() << " " << qwb.y() << " " << qwb.z() << " " << qwb.w() << std::endl;
  }
  f.close();  
}



} // namespace SLIM
