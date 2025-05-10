#ifndef SCENE_GRAPH_H
#define SCENE_GRAPH_H

#include <vector>
#include <stdint.h>
#include <unordered_map>
#include <Eigen/Core>

#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

#include "scene_graph_node.h"
#include "vector_map.h"
#include "factor/graff_coordinate.h"

namespace SLIM {

class SceneGraph {
 public:
  
  typedef std::shared_ptr<SceneGraph> Ptr;

  SceneGraph() = default;

  inline Ptr makeShared() { 
    return Ptr(new SceneGraph(*this)); 
  } 

  inline SceneGraph::Ptr clone() {
    SceneGraph::Ptr graph(new SceneGraph);
    for(int i = 0; i < this->nodes_.size(); ++i) {
      SceneGraphNode node(nodes_[i]);
      graph->nodes_.push_back(node.makeShared());
    }
    graph->block_ = this->block_;
    return graph;
  }

  void transform(const Transform& Tsd);

  void buildFromBlock(const Block::Ptr block);

  void buildSelfAffinity();

 public:
  std::vector<SceneGraphNode::Ptr> nodes_;
  std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> self_affinity_mat_;
  std::vector<std::vector<double>> self_loss_mat_;

  std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> surf_vertex_clouds_;
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr surf_vertices_;
  std::vector<Eigen::MatrixXd> graff_coords_;

  Block::Ptr block_;
};

} // namespace SLIM


#endif