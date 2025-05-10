#ifndef LANDMARK_H
#define LANDMARK_H

#include "factor/line_info.h"
#include "factor/surface_info.h"
#include "observation.h"
#include "frame.h"

namespace SLIM {

using CentroidCloud = pcl::PointCloud<pcl::PointXYZL>;
using CentroidCloudPtr = pcl::PointCloud<pcl::PointXYZL>::Ptr;
using CentroidKdTree = pcl::search::KdTree<pcl::PointXYZL>;
using CentroidKdTreePtr = pcl::search::KdTree<pcl::PointXYZL>::Ptr;
using NodeQuery = std::pair<CentroidCloudPtr, CentroidKdTreePtr>;

class LineLM {
 public:
  typedef std::shared_ptr<LineLM> Ptr;

  LineLM(const uint32_t id, const uint16_t sem_type, const LineInfo& info) 
  : id_(id), sem_type_(sem_type), line_(info), updated_(false) {}

  LineLM(const LineLM::Ptr& other) 
  : id_(other->id()), sem_type_(other->semantic_type()), line_(other->getLineInfo()), 
    obvs_(other->getAllObs()), centroid_(other->centroid()), normal_(other->normal()), updated_(false) {}

  LineLM(const uint32_t id, const uint16_t sem_type, const Eigen::Vector3d& centroid, const Eigen::Vector3d& normal) 
  : id_(id), sem_type_(sem_type), centroid_(centroid), normal_(normal), updated_(false) {
    line_ = LineInfo(centroid_, normal_);
  }

  LineLM(const uint32_t id, Frame::Ptr const& frame, LineOB::Ptr const& ob) {
    id_ = id;
    updated_ = false;
    sem_type_ = ob->semantic_type();
    Transform const Twb = frame->Twb();
    Eigen::Vector3d const point_a = Twb * ob->point_a();
    Eigen::Vector3d const point_b = Twb * ob->point_b();
    centroid_ = 0.5 * (point_a + point_b);
    normal_ = (point_a - point_b).normalized();
    vector2double();
    obvs_.push_back(std::make_pair(frame, ob));
  }

  bool associate(LineOB::Ptr const& ob, Transform const& Twb);

  void addNewOb(Frame::Ptr const& frame, LineOB::Ptr const& ob) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    obvs_.push_back(std::make_pair(frame, ob));
    updated_ = true;
  }

  void addPriorOb(Frame::Ptr const& frame, LineOB::Ptr const& ob) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    prior_obvs_.push_back(std::make_pair(frame, ob));
    updated_ = true;
  }

  void transform(const Transform& Tsd) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    centroid_ = Tsd * centroid_;
    normal_ = Tsd.dcm() * normal_;
    pa_ = Tsd * pa_;
    pb_ = Tsd * pb_;
    vector2double();
  }

  void merge(const LineLM::Ptr& other) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    auto obs = other->getAllObs();
    for(auto &ob: obs) {
      obvs_.push_back(ob);
    }
    updated_ = true;
    solveByObs();
  }

  void solveByObs();

  void setID(const uint32_t id) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    id_ = id;
  }

  uint32_t id() const noexcept {
    return id_;
  }

  uint16_t semantic_type() const {
    return sem_type_;
  }

  const Eigen::Vector3d& centroid() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    return centroid_;
  }

  const Eigen::Vector3d& normal() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    return normal_;
  }

  const Eigen::Matrix3d& sigma() {
    if(!updated_)
      return sigma_;

    std::vector<Eigen::Vector3d> particles;
    for(auto &ob_iter: obvs_) {
      Frame::Ptr const f = ob_iter.first;
      LineOB::Ptr const ob = ob_iter.second;
      particles.push_back(f->Twb() * ob->point_a());
      particles.push_back(f->Twb() * ob->point_b());
    }
    
    Eigen::Vector3d mu{Eigen::Vector3d::Zero()};
    sigma_.setZero();
    for(int i = 0; i < particles.size(); ++i) {
      mu += particles[i];
      sigma_ += particles[i] * particles[i].transpose();
    }
    mu /= particles.size();
    sigma_.noalias() = sigma_ / particles.size() - mu * mu.transpose();
    return sigma_;
  }

  void vector2double() {
    line_ = LineInfo(centroid_, normal_);
  }

  void double2vector();

  Eigen::Vector3d pa() const {
    return pa_;
  }

  Eigen::Vector3d pb() const {
    return pb_;
  }

  LineInfo getLineInfo() const {
    return line_;
  }

  std::vector<std::pair<Frame::Ptr, LineOB::Ptr>> getAllObs() const {
    return obvs_;
  }

  std::vector<std::pair<Frame::Ptr, LineOB::Ptr>> getPriorObs() const {
    return prior_obvs_;
  }

  void resetObs() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    obvs_.clear();
  }

  inline Eigen::Matrix<double, 4, 1>& parameters() {
    return line_.parameters();
  }

  size_t getObSize() const {
    return obvs_.size();
  }

 private:
  static uint32_t factory_id_;
  uint32_t id_;
  uint16_t sem_type_;
  LineInfo line_;
  std::vector<std::pair<Frame::Ptr, LineOB::Ptr>> prior_obvs_;
  std::vector<std::pair<Frame::Ptr, LineOB::Ptr>> obvs_;
  Eigen::Vector3d centroid_, normal_;
  Eigen::Matrix3d sigma_ = Eigen::Matrix3d::Identity();
  Eigen::Vector3d pa_, pb_;
  bool updated_;
  std::mutex data_mutex_;
};


class SurfaceLM {
 public:
  typedef std::shared_ptr<SurfaceLM> Ptr;

  SurfaceLM(const uint16_t sem_type)
  : id_(factory_id_++), sem_type_(sem_type) {}

  SurfaceLM(const uint32_t id, const uint16_t sem_type, const SurfaceInfo& info, const float radius) 
  : id_(id), sem_type_(sem_type), surface_(info), updated_(false), radius_(radius) {}

  SurfaceLM(const SurfaceLM::Ptr& other) 
  : id_(other->id()), sem_type_(other->semantic_type()), radius_(other->getRadius()), surface_(other->getSurfaceInfo()), 
    obvs_(other->getAllObs()), centroid_(other->centroid()), normal_(other->normal()), updated_(false) {}


  SurfaceLM(const uint32_t id, const uint16_t sem_type, const float radius, const Eigen::Vector3d& centroid, const Eigen::Vector3d& normal) 
  : id_(id), sem_type_(sem_type), radius_(radius), centroid_(centroid), normal_(normal), updated_(false) {
    surface_ = SurfaceInfo(centroid_, normal_);
  }

  SurfaceLM(const uint32_t id, Frame::Ptr const& frame, SurfaceOB::Ptr const& ob) {
    id_ = id;
    updated_ = false;
    radius_ = ob->rb();
    sem_type_ = ob->semantic_type();
    Transform const Twb = frame->Twb();
    centroid_ = Twb * ob->centroid();
    normal_ = (Twb.dcm() * ob->normal()).normalized();
    surface_ = SurfaceInfo(centroid_, normal_);
    obvs_.push_back(std::make_pair(frame, ob));
  }

  bool associate(SurfaceOB::Ptr const& ob, Transform const& Twb);

  void addNewOb(Frame::Ptr const& frame, SurfaceOB::Ptr const& ob) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    obvs_.push_back(std::make_pair(frame, ob));
    updated_ = true;
  }

  void addPriorOb(Frame::Ptr const& frame, SurfaceOB::Ptr const& ob) {
    std::unique_lock<std::mutex> lock(data_mutex_);
    prior_obvs_.push_back(std::make_pair(frame, ob));
    updated_ = true;
  }

  void transform(const Transform& Tsd) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    // surface_.transform(Tsd);
    centroid_ = Tsd * centroid_;
    normal_ = Tsd.dcm() * normal_;
    for(auto &vertex: vertices_) {
      vertex = Tsd * vertex;
    }
    vector2double();
  }

  void merge(const SurfaceLM::Ptr& other) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    auto obs = other->getAllObs();
    for(auto &ob: obs) {
      obvs_.push_back(ob);
    }
    updated_ = true;
    // solveByObs();
  }
  
  void solveByObs();
  
  void setID(const uint32_t id) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    id_ = id;
  }

  SurfaceInfo getSurfaceInfo() const {
    return surface_;
  }


  const size_t ObSize() const {
    return obvs_.size();
  }

  const Eigen::Vector3d& centroid() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return centroid_;
  }

  const Eigen::Vector3d& normal() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return normal_;
  }

  const Eigen::Matrix3d& sigma() {
    if(!updated_)
      return sigma_;

    std::vector<Eigen::Vector3d> particles;
    for(auto &ob_iter: obvs_) {
      Frame::Ptr const f = ob_iter.first;
      SurfaceOB::Ptr const ob = ob_iter.second;
      std::vector<Eigen::Vector3d> vertices = ob->vertices();
      for(auto v: vertices) {
        particles.push_back(f->Twb() * v);
      }
    }
    
    Eigen::Vector3d mu{Eigen::Vector3d::Zero()};
    sigma_.setZero();
    for(int i = 0; i < particles.size(); ++i) {
      mu += particles[i];
      sigma_ += particles[i] * particles[i].transpose();
    }
    mu /= particles.size();
    sigma_.noalias() = sigma_ / particles.size() - mu * mu.transpose();
    return sigma_;
  }

  void vector2double() {
    surface_ = SurfaceInfo(centroid_, normal_);
  }

  void double2vector();

  std::vector<Eigen::Vector3d> vertices() const {
    return vertices_;
  }


  uint32_t id() const {
    return id_;
  }

  float getRadius() const {
    return radius_;
  }

  void setRadius(const float r) {
    radius_ = r;
  }


  uint16_t semantic_type() const {
    return sem_type_;
  }

  size_t getObSize() const {
    return obvs_.size();
  }

  std::vector<std::pair<Frame::Ptr, SurfaceOB::Ptr>> getAllObs() const {
    return obvs_;
  }

  std::vector<std::pair<Frame::Ptr, SurfaceOB::Ptr>> getPriorObs() const {
    return prior_obvs_;
  }

  void resetObs() {
    std::unique_lock<std::mutex> lock(data_mutex_);
    obvs_.clear();
  }

  inline Eigen::Matrix<double, 3, 1>& parameters() {
    return surface_.parameters();
  }

 private:
  static uint32_t factory_id_;
  uint32_t id_;
  uint16_t sem_type_;
  float radius_;
  SurfaceInfo surface_;
  std::vector<std::pair<Frame::Ptr, SurfaceOB::Ptr>> prior_obvs_;
  std::vector<std::pair<Frame::Ptr, SurfaceOB::Ptr>> obvs_;
  Eigen::Vector3d centroid_, normal_;
  Eigen::Matrix3d sigma_;
  std::vector<Eigen::Vector3d> vertices_;
  bool updated_ = false;
  std::mutex data_mutex_;
};

class Block {
 public:
  typedef std::shared_ptr<Block> Ptr;

  struct Node {
    typedef std::shared_ptr<Node> Ptr;
    uint32_t id;
    uint16_t sem_type;
    Eigen::Vector3d centroid, normal;
    std::vector<Eigen::Vector3d> vertices;
    double radius;
    
    Node(const uint32_t _id, const uint16_t _sem_type, const Eigen::Vector3d& _centroid, const Eigen::Vector3d& _normal, const std::vector<Eigen::Vector3d>& _vertices, const double _radius)
    : id(_id), sem_type(_sem_type), centroid(_centroid), normal(_normal), vertices(_vertices), radius(_radius) {}
  };

  Block() {
    id_ = factory_id_++;
    initQuery();
  }

  Block(uint32_t id) {
    id_ = id;
    initQuery();
  }

  ~Block() = default;

  void setID(const uint32_t id) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    id_ = id;
  }

  uint32_t getID() const {
    return id_;
  }

  void setHostFrame(const Frame::Ptr& frame) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    host_frame_ = frame;
  }

  void insertLine(const LineLM::Ptr& lm) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    std::vector<Eigen::Vector3d> vertices{lm->pa(), lm->pb()};
    Node::Ptr node(new Node(lm->id(), lm->semantic_type(), lm->centroid(), lm->normal(), vertices, 0.0));
    sem_line_lms_[lm->semantic_type()].insert(std::make_pair(lm->id(), node));
  }

  void insertSurface(const SurfaceLM::Ptr& lm) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    Node::Ptr node(new Node(lm->id(), lm->semantic_type(), lm->centroid(), lm->normal(), lm->vertices(), lm->getRadius()));
    sem_surf_lms_[lm->semantic_type()].insert(std::make_pair(lm->id(), node));
  }

  Frame::Ptr& getHostFrame() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return host_frame_;
  }

  std::map<uint16_t, std::unordered_map<uint32_t, Node::Ptr>>& getSemLines() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_line_lms_;
  }

  std::map<uint16_t, std::unordered_map<uint32_t, Node::Ptr>>& getSemSurfaces() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_surf_lms_;
  }

  std::map<uint16_t, NodeQuery>& getSemLineStruct() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_line_ct_struct_;
  }

  std::map<uint16_t, NodeQuery>& getSemSurfStruct() {
    std::lock_guard<std::mutex> lock(data_mutex_);
    return sem_surf_ct_struct_;
  }

  void initQuery() {
    sem_line_ct_struct_[POLE_ID].first = CentroidCloudPtr(new CentroidCloud());
    sem_surf_ct_struct_[ROAD_ID].first = CentroidCloudPtr(new CentroidCloud());
    sem_surf_ct_struct_[SIDEWALK_ID].first = CentroidCloudPtr(new CentroidCloud());
    sem_surf_ct_struct_[BUILDING_ID].first = CentroidCloudPtr(new CentroidCloud());
    sem_surf_ct_struct_[FENCE_ID].first = CentroidCloudPtr(new CentroidCloud());

    sem_line_ct_struct_[POLE_ID].second = CentroidKdTreePtr(new CentroidKdTree());
    sem_surf_ct_struct_[ROAD_ID].second = CentroidKdTreePtr(new CentroidKdTree());
    sem_surf_ct_struct_[SIDEWALK_ID].second = CentroidKdTreePtr(new CentroidKdTree());
    sem_surf_ct_struct_[BUILDING_ID].second = CentroidKdTreePtr(new CentroidKdTree());    
    sem_surf_ct_struct_[FENCE_ID].second = CentroidKdTreePtr(new CentroidKdTree());    

  }

  void printInfo() {
    printf("Block Info:\n");
    printf("Line Size: %ld\n", sem_line_lms_[POLE_ID].size());
    printf("Surf Size: %ld\n", sem_surf_lms_[ROAD_ID].size());
  }

  void initStruct() {
    sem_line_ct_struct_.clear();
    sem_surf_ct_struct_.clear();

    initQuery();

    for(auto &cls_iter: sem_line_ct_struct_) {
      uint16_t const sem_id = cls_iter.first;
      auto &pair = cls_iter.second;
      auto const& lms = sem_line_lms_[sem_id];
      if(lms.size() == 0) continue;

      pair.first.reset(new CentroidCloud());
      pair.first->reserve(lms.size());
      for(auto lm_iter: lms) {
        Block::Node::Ptr const& node = lm_iter.second;
        pcl::PointXYZL centroid_info;
        centroid_info.getVector3fMap() = node->centroid.cast<float>();
        centroid_info.label = node->id;
        pair.first->push_back(centroid_info);
      }
      pair.second->setInputCloud(pair.first);
    }

    for(auto &cls_iter: sem_surf_ct_struct_) {
      uint16_t const sem_id = cls_iter.first;
      auto &pair = cls_iter.second;
      auto const& lms = sem_surf_lms_[sem_id];
      if(lms.size() == 0) continue;

      pair.first.reset(new CentroidCloud());
      pair.first->reserve(lms.size());
      for(auto lm_iter: lms) {
        Block::Node::Ptr const& node = lm_iter.second;
        pcl::PointXYZL centroid_info;
        centroid_info.getVector3fMap() = node->centroid.cast<float>();
        centroid_info.label = node->id;
        pair.first->push_back(centroid_info);
      }
      pair.second->setInputCloud(pair.first);
    }
  }

  void transform(const Transform& Tsd) {
    std::lock_guard<std::mutex> lock(data_mutex_);
    for(auto &cls_iter: sem_line_lms_) {
      uint16_t cls = cls_iter.first;
      auto lms = cls_iter.second;
      for(auto &lm_iter: lms) {
        lm_iter.second->centroid.noalias() = Tsd * lm_iter.second->centroid;
        lm_iter.second->normal.noalias() = Tsd.dcm() * lm_iter.second->normal;
      }
    }

    for(auto &cls_iter: sem_surf_lms_) {
      uint16_t cls = cls_iter.first;
      auto lms = cls_iter.second;
      for(auto &lm_iter: lms) {
        lm_iter.second->centroid.noalias() = Tsd * lm_iter.second->centroid;
        lm_iter.second->normal.noalias() = Tsd.dcm() * lm_iter.second->normal;
      }
    }

    initStruct();
  }

 private:
  static uint32_t factory_id_;
  uint32_t id_;
  Frame::Ptr host_frame_;
  std::map<uint16_t, std::unordered_map<uint32_t, Node::Ptr>> sem_line_lms_;
  std::map<uint16_t, std::unordered_map<uint32_t, Node::Ptr>> sem_surf_lms_;
  std::map<uint16_t, NodeQuery> sem_line_ct_struct_;
  std::map<uint16_t, NodeQuery> sem_surf_ct_struct_;
  std::mutex data_mutex_;
};

} // namespace SLIM
#endif