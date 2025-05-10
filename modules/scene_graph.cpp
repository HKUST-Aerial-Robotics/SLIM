#include "scene_graph.h"

namespace SLIM {
  
void SceneGraph::transform(const Transform& Tsd) {
  for(auto node: nodes_) {
    node->transform(Tsd);
  }
}

void SceneGraph::buildFromBlock(const Block::Ptr block) {
  int graph_node_id = 0;
  int line_node_num = 0, surf_node_num = 0;
  block_ = block;
  Eigen::Vector3d centroid = block->getHostFrame()->Twb().p();
  nodes_.clear();
  // std::vector<double> dists;
  std::vector<SceneGraphNode::Ptr> line_nodes;
  for(auto &cls_iter: block->getSemLines()) {
    if (line_node_num > 3)
      continue;
    for(auto &lm_iter: cls_iter.second) {
      Block::Node::Ptr nd = lm_iter.second;
      double dist = (nd->centroid - centroid).norm();
      SceneGraphNode::Ptr node = SceneGraphNode::Ptr(
        new SceneGraphNode(graph_node_id++, nd->sem_type, nd->centroid, nd->normal)
      );
      node->AddBlockNode(nd);
      node->dist_ = dist;
      line_nodes.push_back(node);
      line_node_num++;
    }
  }
  std::sort(line_nodes.begin(), line_nodes.end(), [&](const SceneGraphNode::Ptr a, const SceneGraphNode::Ptr b){
    return (a->dist_ < b->dist_);
  });

  if(line_nodes.size() > 20) {
    nodes_ = std::vector<SceneGraphNode::Ptr>{line_nodes.begin(), line_nodes.begin() + 20};
  }
  else {
    nodes_ = line_nodes;
  }
  
  surf_vertices_.reset(new pcl::PointCloud<pcl::PointXYZINormal>());

  std::vector<Block::Node::Ptr> surf_block_nodes;
  for(auto &cls_iter: block->getSemSurfaces()) {
    for(auto &lm_iter: cls_iter.second) {
      Block::Node::Ptr nd = lm_iter.second;
      // if(nd->sem_type != FENCE_ID && nd->sem_type != BUILDING_ID)
      //   continue;
      Eigen::Vector3d const centroid = nd->centroid;
      Eigen::Vector3d const normal = nd->normal;
      pcl::PointXYZINormal vertex;
      vertex.x = centroid(0);
      vertex.y = centroid(1);
      vertex.z = centroid(2);
      vertex.normal_x = normal(0);
      vertex.normal_y = normal(1);
      vertex.normal_z = normal(2);
      vertex.intensity = nd->sem_type;

      surf_vertices_->push_back(vertex);
      surf_block_nodes.push_back(nd);
    }
  }

  SurfaceCluster cluster;
  std::vector<pcl::PointIndices> cluster_res;
  cluster.setCorePointMinPts(5);
  cluster.setAngleTolerance(M_PI/18);
  cluster.setDistTolerance(0.2);
  cluster.setClusterTolerance(5.0);
  cluster.setMinClusterSize(5);
  cluster.setInputCloud(surf_vertices_);
  cluster.extract(cluster_res);  

  for(int i = 0; i < cluster_res.size(); ++i) {
    pcl::PointCloud<pcl::PointXYZINormal> instance(*surf_vertices_, cluster_res[i].indices);  
    uint16_t sem_type = static_cast<uint16_t>(std::round(instance.points.front().intensity));
    pcl::PointXYZINormal const& p0{instance.points.front()};
    Eigen::Vector3d const v0(p0.x, p0.y, p0.z);
    Eigen::Vector3d const n0(p0.normal_x, p0.normal_y, p0.normal_z);
    Eigen::Vector3d mu{v0}, normal{n0};
    Eigen::Matrix3d sigma{Eigen::Matrix3d::Zero()};
    // sigma_.setZero();
    for(int j = 1; j < instance.points.size(); ++j) {
      pcl::PointXYZINormal const& p{instance.points[j]};
      Eigen::Vector3d const v(p.x, p.y, p.z);
      Eigen::Vector3d n(p.normal_x, p.normal_y, p.normal_z);
      if(n.dot(n0) < 0) {
        n = -n;
      }
      mu += v;
      sigma += v * v.transpose();
      normal += n;
    }
    mu /= instance.points.size();
    normal.normalize();
    sigma.noalias() = sigma / instance.points.size() - mu * mu.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma);
    Eigen::Vector3d node_normal = saes.eigenvectors().col(0);

    SceneGraphNode::Ptr node = SceneGraphNode::Ptr(
      new SceneGraphNode(graph_node_id, sem_type, mu, normal)
    );
    node->setSigma(sigma);
    for(auto id: cluster_res[i].indices) {
      node->AddBlockNode(surf_block_nodes[id]);
    }
    nodes_.push_back(node);
    graph_node_id++;
    surf_node_num++;
  }
}

void SceneGraph::buildSelfAffinity() {
  self_loss_mat_.resize(nodes_.size());
  for(int i = 0; i < nodes_.size(); ++i) {
    for(int j = 0; j < nodes_.size(); ++j) {
      if (i == j) 
        continue;
        
      auto const& node_i = nodes_[i];
      auto const& node_j = nodes_[j];

      Eigen::MatrixXd gi = node_i->graff_coord_;
      Eigen::MatrixXd gj = node_j->graff_coord_;

      const Eigen::Vector3d b0i = gi.topRightCorner(3, 1);
      gi.topRightCorner(3, 1).setZero();
      gj.topRightCorner(3, 1) -= b0i;
      const Eigen::Vector3d b0j = gj.topRightCorner(3, 1);
      gj.rightCols<1>() /= std::sqrt(1 + b0j.squaredNorm());
      
      auto sigma = Eigen::JacobiSVD<Eigen::MatrixXd>(gi.transpose() * gj).singularValues();
      double metric = 0.0;
      for(int i = 0; i < sigma.rows(); ++i) {
        // double theta = std::tan(std::acos(sigma(i)));
        // double theta = std::acos(sigma(i));
        double theta = sigma(i);
        metric += theta*theta;
      }
      metric = std::sqrt(metric);
      self_affinity_mat_[i][j] = sigma.norm();
      // self_affinity_mat_[i][j] = metric;
    }
  }
}

} // namespace SLIM
