#ifndef SEMANTIC_GRAPH_NODE_H
#define SEMANTIC_GRAPH_NODE_H

#include "observation.h"
#include "landmark.h"
#include "dbscan/surface_cluster.h"
#include "factor/math_utility.h"
#include "factor/graff_coordinate.h"

namespace SLIM {

class SceneGraphNode {
 public:
  typedef std::shared_ptr<SceneGraphNode> Ptr;

  enum GeoType {
    LINE = 1,
    SURFACE = 2,
    NONE = 3,
  };

  SceneGraphNode()
  : id_(0), semantic_type_(NONE_FEATURE_ID), node_value_(Eigen::Vector3d::Zero()), node_normal_(Eigen::Vector3d::Zero()) {
    SetGeoType(NONE_FEATURE_ID);
  }

  SceneGraphNode(const uint32_t id, const uint16_t semantic_type, const Eigen::Vector3d& node_value, const Eigen::Vector3d& node_normal)
  : id_(id), semantic_type_(semantic_type), node_value_(node_value), node_normal_(node_normal) {
    SetGeoType(semantic_type);
    setGraffCoord();
    if(geometry_type_ == SURFACE) {
      double d = -node_normal_.dot(node_value_);
    }
  }

  SceneGraphNode(const SceneGraphNode::Ptr other)
  : id_(other->id_), semantic_type_(other->semantic_type_), geometry_type_(other->geometry_type_),
    node_value_(other->node_value_), node_normal_(other->node_normal_),
    vec_line_feature_ptr_(other->vec_line_feature_ptr_), vec_surf_feature_ptr_(other->vec_surf_feature_ptr_),
    vec_line_lms_(other->vec_line_lms_), vec_surf_lms_(other->vec_surf_lms_), vec_nodes_(other->vec_nodes_) {
    setGraffCoord();
  }

  inline Ptr makeShared() { 
    return Ptr(new SceneGraphNode(*this)); 
  } 

  void SetGeoType(const uint16_t semantic_type) {
    switch (semantic_type_)
    {
    case NONE_FEATURE_ID:
      geometry_type_ = NONE;
      break;
    
    case POLE_ID:
      geometry_type_ = LINE;
      break;
    
    case TRAFFIC_SIGN_ID:
      geometry_type_ = SURFACE;
      break;
    
    case ROAD_ID:
      // geometry_type_ = SURFACE;
      geometry_type_ = SURFACE;
      break;

    case SIDEWALK_ID:
      // geometry_type_ = SURFACE;
      geometry_type_ = SURFACE;
      break;
    
    case BUILDING_ID:
      // geometry_type_ = SURFACE;
      geometry_type_ = SURFACE;
      break;

    case FENCE_ID:
      // geometry_type_ = SURFACE;
      geometry_type_ = SURFACE;
      break;
    
    default:
      geometry_type_ = NONE;
      break;
    }
  }

  void setGraffCoord() {
    if(geometry_type_ == GeoType::LINE) {
      graff_coord_ = GraffLine(node_normal_, node_value_ / 40.0).get();
      line_info_ = LineInfo(node_value_ / 40.0, node_normal_);
    }
    else if(geometry_type_ == GeoType::SURFACE) {
      graff_coord_ = GraffSurface(SurfaceInfo(node_value_, node_normal_).subspace(), node_value_ / 40.0).get();
      surface_info_ = SurfaceInfo(node_value_ / 40.0, node_normal_);
    }
  }

  void setSigma(const Eigen::Matrix3d& sigma) {
    sigma_ = sigma;
  }
  
  uint32_t id() const {
    return id_;
  }

  uint16_t semantic_type() const {
    return semantic_type_;
  }

  GeoType geometry_type() const {
    return geometry_type_;
  }

  Eigen::Vector3d node_value() const {
    return node_value_;
  }

  Eigen::Vector3d normal_value() const {
    return node_normal_;
  }

  void AddLineFeaturePtr(const LineOB::Ptr& ptr) {
    vec_line_feature_ptr_.push_back(ptr);
  }

  void AddSurfFeaturePtr(const SurfaceOB::Ptr& ptr) {
    vec_surf_feature_ptr_.push_back(ptr);
  }

  void AddLineLandmark(const LineLM::Ptr& ptr) {
    vec_line_lms_.push_back(ptr);
  }

  void AddSurfLandmark(const SurfaceLM::Ptr& ptr) {
    vec_surf_lms_.push_back(ptr);
  }

  void AddBlockNode(const Block::Node::Ptr& ptr) {
    vec_nodes_.push_back(ptr);
  }

  void transform(const Transform& Tsd) {
    node_value_ = Tsd * node_value_;
    node_normal_ = Tsd.dcm() * node_normal_;
  }

 public:
  uint32_t id_; // must less than 2^21
  uint16_t semantic_type_;
  GeoType geometry_type_;
  Eigen::Vector3d node_value_;
  Eigen::Vector3d node_normal_;
  Eigen::Matrix3d sigma_;
  double dist_;
  std::vector<LineOB::Ptr> vec_line_feature_ptr_;
  std::vector<SurfaceOB::Ptr> vec_surf_feature_ptr_;

  std::vector<LineLM::Ptr> vec_line_lms_;
  std::vector<SurfaceLM::Ptr> vec_surf_lms_;

  std::vector<Block::Node::Ptr> vec_nodes_;
  Eigen::MatrixXd graff_coord_;
  LineInfo line_info_;
  SurfaceInfo surface_info_;
};
} // namespace SLIM



#endif