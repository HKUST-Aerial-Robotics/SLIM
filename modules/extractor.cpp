#include "extractor.h"
#include <pcl/common/transforms.h>
#include <pcl/visualization/point_cloud_color_handlers.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace SLIM {

using std::atan2;
using std::cos;
using std::sin;

cv::Vec3b ComputeJetColorMap(double v, double vmin, double vmax) 
{
  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  return cv::Vec3b((uint8_t)(255 * db), (uint8_t)(255 * dg), (uint8_t)(255 * dr));
}

void solveLine(const pcl::PointCloud<pcl::PointXYZI>& cloud, Eigen::Vector3f& mu, Eigen::Vector3f& normal, Eigen::Vector3f& lambda, Eigen::Matrix3f& sigma) {
  // Eigen::Matrix3f sigma{Eigen::Matrix3f::Zero()};
  mu.setZero();
  sigma.setZero();
  solveCovMat(cloud, mu, sigma);
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> saes(sigma);
  lambda = saes.eigenvalues();
  normal = saes.eigenvectors().col(2).normalized();
}

void VLPExtractor::setResolution(const int w, const int h) {
  img_.resize(h);
  for(int row = 0; row < h; ++row) {
    img_[row].resize(w);
  }
  w_ = w;
  h_ = h;
}

void VLPExtractor::extract(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  img_.clear();
  img_.resize(h_);
  for(int row = 0; row < h_; ++row) {
    img_[row].reserve(w_);
  }

  TicToc timer;
  int prev_quadrant = 0;
  int ring = 0;
  for(int i = 0; i < cloud->size(); ++i) {
    pcl::PointXYZI pt;
    pt = cloud->points[i];
    int quadrant = getQuadrant(pt);
    if((quadrant == 1) && (prev_quadrant == 4)) {
      ring ++;
    }
    if(ring == h_)
      break;
    pt.intensity = (ring * 20) % 255;
    Eigen::Vector3f pos = pt.getVector3fMap();
    img_[ring].push_back(ScanPoint(pos, 0));
    prev_quadrant = quadrant;
  }
  std::cout << "split time: " << timer.toc() << std::endl;
  timer.tic();
  surf_cloud_.clear();
  raw_surf_cloud_.clear();
  pole_cloud_.clear();

  // for(int row = 0; row < h_; ++row) {
  //   int size = img_[row].size() / 2;
  //   for(int col = 0; col < size; ++col) {
  //     img_[row][col] = img_[row][col * 2];
  //   }
  //   img_[row].resize(size);
  // }

  const float edge_thres = 0.5;
  for(int row = 0; row < h_; ++row) {
    for(int col = kNumCurvSize; col < img_[row].size() - kNumCurvSize; ++col) {
      std::vector<int> nb_left, nb_right;
      for(int bias = -1; col+bias >= 0; --bias) {
        if(img_[row][col+bias].null)
          continue;
  
        nb_left.push_back(col+bias);
        if(nb_left.size() == kNumCurvSize)
          break;
      }
      for(int bias = 1; col+bias <= img_[row].size(); ++bias) {
        if(img_[row][col+bias].null)
          continue;
  
        nb_right.push_back(col+bias);
        if(nb_right.size() == kNumCurvSize)
          break;
      }
      int cnt = std::min(nb_left.size(), nb_right.size());

      if(cnt >= kNumCurvSize) {

        std::vector<float> ed(kNumCurvSize);
        for(int i = 0; i < kNumCurvSize; ++i) {
          float dist = ((img_[row][col].pos - img_[row][nb_left[i]].pos).cross(img_[row][col].pos - img_[row][nb_right[i]].pos)).norm() / (img_[row][nb_left[i]].pos - img_[row][nb_right[i]].pos).norm();
          ed[i] = dist;
        }

        if(*std::max_element(ed.begin(), ed.end()) < 0.2) {
          img_[row][col].attribute = PointAttribute::SURFACE;
          pcl::PointXYZI point;
          point.getVector3fMap() = img_[row][col].pos;
          surf_cloud_.push_back(point);
          raw_surf_cloud_.push_back(point);
        }
      }

      if(img_[row][col].range > 100.0f) {
        continue;
      }

      if(!img_[row][col-1].null && !img_[row][col+1].null) { // has left and right valid neighbors
        float lg = img_[row][col-1].range - img_[row][col].range; // left diff
        float rg = img_[row][col+1].range - img_[row][col].range; // right diff
        img_[row][col].grad = std::max(lg, rg);
        float dist = ((img_[row][col].pos - img_[row][col-1].pos).cross(img_[row][col].pos - img_[row][col+1].pos)).norm() / (img_[row][col-1].pos - img_[row][col+1].pos).norm();
        if(lg > edge_thres && rg > edge_thres && dist > 0.2) {
          img_[row][col].attribute = PointAttribute::POLE;
        }
        else if(lg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_LEFT;
        }
        else if(rg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_RIGHT;
        }
      }
      else if(!img_[row][col-1].null && img_[row][col+1].null) {  // has left valid neighbors
        float lg = img_[row][col-1].range - img_[row][col].range;
        img_[row][col].grad = std::max(lg, 0.0f);
        if(lg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_LEFT;
        }
      }
      else if(img_[row][col-1].null && !img_[row][col+1].null) {  // has right valid neighbors
        float rg = img_[row][col+1].range - img_[row][col].range;
        img_[row][col].grad = std::max(rg, 0.0f);
        if(rg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_RIGHT;
        }
      }
      else {
        img_[row][col].grad = 0.0f;
      }
    }
  }
  std::cout << "preprocess time: " << timer.toc() << std::endl;

  timer.tic();
  extractPole();
  std::cout << "extract pole time: " << timer.toc() << std::endl;

  timer.tic();
  extractSurface();
  std::cout << "extract surface time: " << timer.toc() << std::endl;
}

void VLPExtractor::extract(const pcl::PointCloud<OusterPoint>::Ptr cloud) {
  img_.clear();
  img_.resize(h_);
  for(int row = 0; row < h_; ++row) {
    img_[row].reserve(2000);
  }
  for(int i = 0; i < cloud->size(); ++i) {
    Eigen::Vector3f point(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
    if(point.norm() > 0.1) {
      img_[cloud->points[i].ring].push_back(ScanPoint(point, 0));
    }
  }

  for(int row = 0; row < h_; ++row) {
    img_[row].shrink_to_fit();
  }

  surf_cloud_.clear();
  raw_surf_cloud_.clear();
  pole_cloud_.clear();
  
  TicToc timer;
  const float edge_thres = 0.5;
  for(int row = 0; row < h_; ++row) {
    for(int col = kNumCurvSize; col < img_[row].size() - kNumCurvSize; ++col) {
      std::vector<int> nb_left, nb_right;
      for(int bias = -1; col+bias >= 0; --bias) {
        if(img_[row][col+bias].null)
          continue;
  
        nb_left.push_back(col+bias);
        if(nb_left.size() == kNumCurvSize)
          break;
      }
      for(int bias = 1; col+bias <= img_[row].size(); ++bias) {
        if(img_[row][col+bias].null)
          continue;
  
        nb_right.push_back(col+bias);
        if(nb_right.size() == kNumCurvSize)
          break;
      }
      int cnt = std::min(nb_left.size(), nb_right.size());

      if(cnt >= kNumCurvSize) {

        std::vector<float> ed(kNumCurvSize);
        for(int i = 0; i < kNumCurvSize; ++i) {
          float dist = ((img_[row][col].pos - img_[row][nb_left[i]].pos).cross(img_[row][col].pos - img_[row][nb_right[i]].pos)).norm() / (img_[row][nb_left[i]].pos - img_[row][nb_right[i]].pos).norm();
          ed[i] = dist;
        }

        if(*std::max_element(ed.begin(), ed.end()) < 0.2) {
          img_[row][col].attribute = PointAttribute::SURFACE;
          pcl::PointXYZI point;
          point.getVector3fMap() = img_[row][col].pos;
          surf_cloud_.push_back(point);
          raw_surf_cloud_.push_back(point);
        }
      }

      if(img_[row][col].range > 100.0f) {
        continue;
      }

      if(!img_[row][col-1].null && !img_[row][col+1].null) { // has left and right valid neighbors
        float lg = img_[row][col-1].range - img_[row][col].range; // left diff
        float rg = img_[row][col+1].range - img_[row][col].range; // right diff
        img_[row][col].grad = std::max(lg, rg);
        float dist = ((img_[row][col].pos - img_[row][col-1].pos).cross(img_[row][col].pos - img_[row][col+1].pos)).norm() / (img_[row][col-1].pos - img_[row][col+1].pos).norm();
        if(lg > edge_thres && rg > edge_thres && dist > 0.2) {
          img_[row][col].attribute = PointAttribute::POLE;
        }
        else if(lg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_LEFT;
        }
        else if(rg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_RIGHT;
        }
      }
      else if(!img_[row][col-1].null && img_[row][col+1].null) {  // has left valid neighbors
        float lg = img_[row][col-1].range - img_[row][col].range;
        img_[row][col].grad = std::max(lg, 0.0f);
        if(lg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_LEFT;
        }
      }
      else if(img_[row][col-1].null && !img_[row][col+1].null) {  // has right valid neighbors
        float rg = img_[row][col+1].range - img_[row][col].range;
        img_[row][col].grad = std::max(rg, 0.0f);
        if(rg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_RIGHT;
        }
      }
      else {
        img_[row][col].grad = 0.0f;
      }
    }
  }
  extractPole();
  extractSurface();
}

bool detectLocalSurf(const std::vector<ScanPoint>& pset) {
  Eigen::Vector3f mu{Eigen::Vector3f::Zero()};
  Eigen::Matrix3f sigma{Eigen::Matrix3f::Zero()};
  for(int i = 0; i < pset.size(); ++i) {
    mu += pset[i].pos;
    sigma += pset[i].pos * pset[i].pos.transpose();
  }
  mu /= pset.size();
  sigma.noalias() = sigma / pset.size() - mu * mu.transpose();


  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> saes(sigma);
  
  Eigen::Vector3f lambda = saes.eigenvalues();
  Eigen::Vector3f normal = saes.eigenvectors().col(0);
  if(lambda(0) > 0.1 * lambda(1))
    return false;
  

  double local_radius = std::sqrt(lambda(2) * 2);
  int inlier_num = 0;
  for(int i = 0; i < pset.size(); ++i) {
    auto dist = normal.transpose() * (pset[i].pos - mu);
    if(std::abs(dist) < 0.1 * local_radius) {
      inlier_num++;
    }
  }
  if(inlier_num == pset.size())
    return true;

  return false;
}


void VLPExtractor::extract(const std::vector<std::vector<ScanPoint>>& img) {
  surf_cloud_.clear();
  raw_surf_cloud_.clear();
  pole_cloud_.clear();
  
  img_ = img;
  const float edge_thres = 0.5;
  for(int row = 0; row < h_; ++row) {
    for(int col = kNumCurvSize; col < img_[row].size() - kNumCurvSize; ++col) {
      std::vector<int> nb_left, nb_right;
      for(int bias = -1; col+bias >= 0; --bias) {
        if(img_[row][col+bias].null)
          continue;
  
        nb_left.push_back(col+bias);
        if(nb_left.size() == kNumCurvSize)
          break;
      }
      for(int bias = 1; col+bias <= img_[row].size(); ++bias) {
        if(img_[row][col+bias].null)
          continue;
  
        nb_right.push_back(col+bias);
        if(nb_right.size() == kNumCurvSize)
          break;
      }
      int cnt = std::min(nb_left.size(), nb_right.size());

      if(cnt >= kNumCurvSize) {

        std::vector<float> ed(kNumCurvSize);
        for(int i = 0; i < kNumCurvSize; ++i) {
          float dist = ((img_[row][col].pos - img_[row][nb_left[i]].pos).cross(img_[row][col].pos - img_[row][nb_right[i]].pos)).norm() / (img_[row][nb_left[i]].pos - img_[row][nb_right[i]].pos).norm();
          ed[i] = dist;
        }

        if(*std::max_element(ed.begin(), ed.end()) < 0.2) {
          img_[row][col].attribute = PointAttribute::SURFACE;
          pcl::PointXYZI point;
          point.getVector3fMap() = img_[row][col].pos;
          surf_cloud_.push_back(point);
          raw_surf_cloud_.push_back(point);
          continue;
        }
      }

      if(img_[row][col].range > 100.0f) {
        continue;
      }

      if(!img_[row][col-1].null && !img_[row][col+1].null) { // has left and right valid neighbors
        float lg = img_[row][col-1].range - img_[row][col].range; // left diff
        float rg = img_[row][col+1].range - img_[row][col].range; // right diff
        img_[row][col].grad = std::max(lg, rg);
        float dist = ((img_[row][col].pos - img_[row][col-1].pos).cross(img_[row][col].pos - img_[row][col+1].pos)).norm() / (img_[row][col-1].pos - img_[row][col+1].pos).norm();
        if(lg > edge_thres && rg > edge_thres && dist > 0.3) {
          img_[row][col].attribute = PointAttribute::POLE;
        }
        else if(lg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_LEFT;
        }
        else if(rg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_RIGHT;
        }
      }
      else if(!img_[row][col-1].null && img_[row][col+1].null) {  // has left valid neighbors
        float lg = img_[row][col-1].range - img_[row][col].range;
        img_[row][col].grad = std::max(lg, 0.0f);
        if(lg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_LEFT;
        }
      }
      else if(img_[row][col-1].null && !img_[row][col+1].null) {  // has right valid neighbors
        float rg = img_[row][col+1].range - img_[row][col].range;
        img_[row][col].grad = std::max(rg, 0.0f);
        if(rg > edge_thres) {
          img_[row][col].attribute = PointAttribute::EDGE_RIGHT;
        }
      }
      else {
        img_[row][col].grad = 0.0f;
      }
    }
  }
  
  extractPole();
  // TicToc timer;
  extractSurface();
  // std::cout << "surface time: " << timer.toc() << std::endl;
}

void VLPExtractor::extractPole() {
  TicToc timer;
  
  for(int row = 0; row < h_; ++row) {
    int start_index = -1;
    std::vector<int> pole_indices;
    // find points with large gradient
    for(int col = kNumCurvSize; col < img_[row].size() - kNumCurvSize; ++col) {
      if(start_index == -1) {
        if(img_[row][col].attribute == PointAttribute::EDGE_LEFT) {
          start_index = col;
          pole_indices.clear();
          pole_indices.push_back(col);
          continue;
        }      
        else if(img_[row][col].attribute == PointAttribute::EDGE_RIGHT) {
          continue;
        }  
        else {
          continue;
        }
      }
      else {
        if(img_[row][col].attribute == PointAttribute::EDGE_LEFT) {
          start_index = col;
          pole_indices.clear();
          pole_indices.push_back(col);
          continue;
        }    
        else if(img_[row][col].attribute == PointAttribute::EDGE_RIGHT) {
          if((img_[row][start_index].pos - img_[row][col].pos).norm() < 0.2) {
            img_[row][start_index].attribute = PointAttribute::POLE;
            for(auto index: pole_indices) {
              img_[row][index].attribute = PointAttribute::POLE;
            }
            img_[row][col].attribute = PointAttribute::POLE;
            start_index = -1;
            pole_indices.clear();
          }
          else {
            start_index = -1;
            pole_indices.clear();
          }
          continue;
        } 
        else if(img_[row][col].attribute == PointAttribute::POLE) {
          start_index = -1;
          pole_indices.clear();
          continue;
        } 
        else {
          pole_indices.push_back(col);
          continue;
        }
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZI> pole_cloud;
  for(int row = 0; row < img_.size(); ++row) {
    for(int col = 0; col < img_[row].size(); ++col) {
      float grad = img_[row][col].grad;
      if(img_[row][col].attribute == PointAttribute::POLE) {
        pcl::PointXYZI point;
        point.getVector3fMap() = img_[row][col].pos;
        point.intensity = 50;
        pole_cloud.push_back(point);
      }
    }
  }

  pcl::EuclideanClusterExtraction<pcl::PointXYZI> cluster;
  cluster.setClusterTolerance(1.0);
  cluster.setMinClusterSize(10);
  cluster.setInputCloud(pole_cloud.makeShared());
  std::vector<pcl::PointIndices> cluster_res;
  cluster.extract(cluster_res);

  pole_cloud_.clear();
  vec_line_feature_.clear();
  srand((unsigned)time(NULL));

  int valid_cluster_num = 0;
  for(int i = 0; i < cluster_res.size(); ++i) {
    
    pcl::PointCloud<pcl::PointXYZI> instance(pole_cloud, cluster_res[i].indices);
    // check PCA
    Eigen::Vector3f mu, normal, lambda;
    Eigen::Matrix3f sigma;
    solveLine(instance, mu, normal, lambda, sigma);
 
    if(lambda(1) > 0.05 * lambda(2)) {
      continue;
    }

    double angle = std::acos(normal.dot(Eigen::Vector3f(0, 0, 1))) * 180 / M_PI;
    if(angle > 20 && angle < 160)
      continue;

    Eigen::Vector3d point_a = (mu + normal * std::sqrt(lambda(2) * 2)).cast<double>();
    Eigen::Vector3d point_b = (mu - normal * std::sqrt(lambda(2) * 2)).cast<double>();
    double length = (point_b - point_a).norm();
    if (length < 1.0)
      continue;
    
    pcl::PointCloud<pcl::PointXYZI> inlier_instance;
    for(auto point: instance.points) {
      Eigen::Vector3f pt = point.getVector3fMap();
      auto dist = ((Eigen::Matrix3f::Identity() - normal * normal.transpose()) * (pt - mu)).norm();
      if(dist < 0.1 * length) {
        inlier_instance.push_back(point);
      }
    }

    if (inlier_instance.size() < 10) {
      continue;
    }
    solveLine(inlier_instance, mu, normal, lambda, sigma);
    point_a = (mu + normal * std::sqrt(lambda(2) * 2)).cast<double>();
    point_b = (mu - normal * std::sqrt(lambda(2) * 2)).cast<double>();
    // Eigen::Matrix4d sqrt_info = Eigen::Matrix4d::Identity() * 1 / (std::sqrt(lambda(1) * 2) + 0.3);
    Eigen::Matrix4d sqrt_info = Eigen::Matrix4d::Identity() * 1 / (30.0);
    
    // if(!is_consist_line)
    //   continue;

    LineOB::Ptr feature = LineOB::Ptr(new LineOB(POLE_ID, point_a, point_b));
    feature->setSqrtInfo(sqrt_info);
    vec_line_feature_.push_back(feature);

    int intensity = rand() % 10000;
    for(auto &point: inlier_instance.points) {
      point.intensity = intensity;
    }
    pole_cloud_ += inlier_instance;
    valid_cluster_num++;
  }
}


void VLPExtractor::extractSurface() {

  ovmap_.clear();
  Transform Twl;
  pcl::VoxelGrid<pcl::PointXYZI> vf;
  vf.setLeafSize(0.1, 0.1, 0.1);
  vf.setInputCloud(surf_cloud_.makeShared());
  vf.filter(surf_cloud_);

  // allocate hash octree and cut the cloud
  const double max_res = 16.0;
  cutCloud(surf_cloud_, Twb_, 0, Eigen::Vector3f(max_res, max_res, max_res), ovmap_);
  surf_cloud_.clear();

  Transform Tbw = Twb_.inverse();
  Eigen::Matrix3d Rbw = Tbw.dcm();
  Eigen::Vector3d tbw = Tbw.p();
  vec_surf_feature_.clear();

  srand((unsigned)time(NULL));
  std::vector<pcl::PointCloud<pcl::PointXYZI>> leaf_nodes;
  surf_voxels_.clear();
  for(auto iter = ovmap_.begin(); iter != ovmap_.end(); ++iter) {
    iter->second->parse();
    std::vector<OctoVoxel> vec_voxel;
    iter->second->getLeafVoxel(vec_voxel);

    for(auto voxel: vec_voxel) {
      Eigen::Vector3d centroid = Tbw * voxel.center_.cast<double>();
      Eigen::Vector3d normal = Rbw * voxel.normal_.cast<double>();
      Eigen::Matrix3d sigma = Rbw * voxel.sigma_.cast<double>() * Rbw.transpose();
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(sigma);

      std::vector<Eigen::Vector3d> vertices(3);
      Eigen::Vector3d lambda = saes.eigenvalues();
      Eigen::Matrix3d umat = saes.eigenvectors();

      Eigen::Matrix3d sqrt_info = Eigen::Matrix3d::Identity() * 1 / (30.0);
      if (sqrt_info.array().isNaN().any()) {
        continue;
      }    

      float ra = std::sqrt(lambda(2) * 2);
      float rb = std::sqrt(lambda(1) * 2);
      vertices[0] = centroid + umat.col(2) * ra;
      vertices[1] = centroid - umat.col(2) * ra * 0.5 + umat.col(1) * rb * 0.5;
      vertices[2] = centroid - umat.col(2) * ra * 0.5 - umat.col(1) * rb * 0.5;
      
      SurfaceOB::Ptr surface_ob = SurfaceOB::Ptr(new SurfaceOB(ROAD_ID, centroid, normal, vertices, ra, rb));
      surface_ob->setSqrtInfo(sqrt_info);
      vec_surf_feature_.push_back(surface_ob);
    }    
    surf_voxels_.insert(surf_voxels_.end(), vec_voxel.begin(), vec_voxel.end());
  }
  assert(leaf_nodes.size() == surf_voxels_.size());

  for(auto node: surf_voxels_) {
    int intensity = rand() % 10000;
    for(auto &point: node.cloud_) {
      point.getVector3fMap() = Rbw.cast<float>() * point.getVector3fMap() + tbw.cast<float>();
      point.intensity = intensity;
    }
    surf_cloud_ += node.cloud_;
  }
}

void VLPExtractor::extractPole(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int scan_line) {
  std::vector<std::vector<ScanPoint>> range_img;
  range_img.resize(scan_line);
  for(int row = 0; row < scan_line; ++row) {
    range_img[row].reserve(2000);
  }
  for(int i = 0; i < cloud->size(); ++i) {
    Eigen::Vector3f point(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
    if(point.norm() > 1.0) {
      float yaw = std::atan2(point.y(), point.x());
      // yaw += M_PI/2;
      range_img[(uint32_t)cloud->points[i].intensity].push_back(ScanPoint(point, yaw));
    }
  }
  for(int row = 0; row < scan_line; ++row) {
    range_img[row].shrink_to_fit();
    std::sort(range_img[row].begin(), range_img[row].end(), [&](const ScanPoint& p1, const ScanPoint& p2) {
      return p1.yaw > p2.yaw;
    });
    // std::cout << "row: " << row << " size: " << range_img[row].size() << std::endl;
  }

  // pcl::visualization::PCLVisualizer viewer("pcl_viewer");
  // pcl::PointCloud<pcl::PointXYZRGB>::Ptr vis_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  // for(int row = 0; row < scan_line; row++) {
  //   for(int col = 0; col < range_img[row].size(); col++) {
  //     float ratio = (float)col / range_img[row].size();
  //     // if(ratio > 1.0)
  //     //   ratio = 1.0;

  //     pcl::PointXYZRGB point;
  //     point.getVector3fMap() = range_img[row][col].pos;
  //     // point.r = ratio * 255;
  //     // point.g = ratio * 255;
  //     // point.b = ratio * 255;
  //     point.r = random_color_vec[row % random_color_vec.size()][2];
  //     point.g = random_color_vec[row % random_color_vec.size()][1];
  //     point.b = random_color_vec[row % random_color_vec.size()][0];

  //     vis_cloud->push_back(point);
  //   }
  //   // std::cout << "row: " << row << " size: " << range_img[row].size() << std::endl;
  // }
  // viewer.addPointCloud(vis_cloud, "frame");
  // viewer.spin();


  const float edge_thres = 0.5;
  for(int row = 0; row < scan_line; ++row) {
    for(int col = kNumCurvSize; col < range_img[row].size() - kNumCurvSize; ++col) {
      std::vector<double> angles;
      for(int bias = 1; bias <= kNumCurvSize; ++bias) {
        Eigen::Vector3f veca = (range_img[row][col-bias].pos - range_img[row][col].pos).normalized();
        Eigen::Vector3f vecb = (range_img[row][col+bias].pos - range_img[row][col].pos).normalized();
        double angle = std::acos(veca.dot(vecb)) * 180.0f / M_PI;
        angles.push_back(angle);
      }

      Eigen::Vector3f cl = range_img[row][col-1].pos, cr = range_img[row][col+1].pos, cm = range_img[row][col].pos;
      float dist = ((cm - cl).cross(cm - cr)).norm() / (cl - cr).norm();
      if(dist < 1.0)
        continue;

      // if(*std::max_element(angles.begin(), angles.end()) < 5.0) {
      //   range_img[row][col].attribute = PointAttribute::SURFACE;
      //   pcl::PointXYZI point;
      //   point.getVector3fMap() = range_img[row][col].pos;
      //   surf_cloud_.push_back(point);
      //   raw_surf_cloud_.push_back(point);
      //   continue;
      // }
      // std::cout << "range: " << range_img[row][col].range << std::endl;
      if(range_img[row][col].range > 50.0f) {
        continue;
      }

      float lg = range_img[row][col-1].range - range_img[row][col].range; // left diff
      float rg = range_img[row][col+1].range - range_img[row][col].range; // right diff
      // std::cout << "lg: " << lg << " rg: " << rg << std::endl;
      range_img[row][col].grad = std::max(lg, rg);
      if(lg > edge_thres && rg > edge_thres) {
        range_img[row][col].attribute = PointAttribute::POLE;
      }
      else if(lg > edge_thres) {
        range_img[row][col].attribute = PointAttribute::EDGE_LEFT;
      }
      else if(rg > edge_thres) {
        range_img[row][col].attribute = PointAttribute::EDGE_RIGHT;
      }    
    }
  }

  for(int row = 0; row < scan_line; ++row) {
    int start_index = -1;
    std::vector<int> pole_indices;
    // find points with large gradient
    for(int col = kNumCurvSize; col < range_img[row].size() - kNumCurvSize; ++col) {
      if(start_index == -1) {
        if(range_img[row][col].attribute == PointAttribute::EDGE_LEFT) {
          start_index = col;
          pole_indices.clear();
          pole_indices.push_back(col);
          continue;
        }      
        else if(range_img[row][col].attribute == PointAttribute::EDGE_RIGHT) {
          continue;
        }  
        else {
          continue;
        }
      }
      else {
        if(range_img[row][col].attribute == PointAttribute::EDGE_LEFT) {
          start_index = col;
          pole_indices.clear();
          pole_indices.push_back(col);
          continue;
        }    
        else if(range_img[row][col].attribute == PointAttribute::EDGE_RIGHT) {
          if((range_img[row][start_index].pos - range_img[row][col].pos).norm() < 0.3) {
            range_img[row][start_index].attribute = PointAttribute::POLE;
            for(auto index: pole_indices) {
              range_img[row][index].attribute = PointAttribute::POLE;
            }
            range_img[row][col].attribute = PointAttribute::POLE;
            start_index = -1;
            pole_indices.clear();
          }
          else {
            start_index = -1;
            pole_indices.clear();
          }
          continue;
        } 
        else if(range_img[row][col].attribute == PointAttribute::POLE) {
          start_index = -1;
          pole_indices.clear();
          continue;
        } 
        else {
          pole_indices.push_back(col);
          continue;
        }
      }
    }
  }

  pcl::PointCloud<pcl::PointXYZI> pole_cloud;
  pole_cloud_.clear();
  for(int row = 0; row < range_img.size(); ++row) {
    for(int col = 0; col < range_img[row].size(); ++col) {
      float grad = range_img[row][col].grad;
      // if(range_img[row][col].attribute == PointAttribute::POLE || \
      //    range_img[row][col].attribute == PointAttribute::EDGE_LEFT || \
      //    range_img[row][col].attribute == PointAttribute::EDGE_RIGHT) {
      if(range_img[row][col].attribute == PointAttribute::POLE) {
        pcl::PointXYZI point;
        point.getVector3fMap() = range_img[row][col].pos;
        point.intensity = 50;
        pole_cloud.push_back(point);
      }
    }
  }

  pcl::EuclideanClusterExtraction<pcl::PointXYZI> cluster;
  cluster.setClusterTolerance(1.0);
  cluster.setMinClusterSize(10);
  cluster.setInputCloud(pole_cloud.makeShared());
  std::vector<pcl::PointIndices> cluster_res;
  cluster.extract(cluster_res);

  pole_cloud_.clear();
  vec_line_feature_.clear();
  srand((unsigned)time(NULL));

  int valid_cluster_num = 0;
  for(int i = 0; i < cluster_res.size(); ++i) {
    
    pcl::PointCloud<pcl::PointXYZI> instance(pole_cloud, cluster_res[i].indices);
    // check PCA
    Eigen::Vector3f mu, normal, lambda;
    Eigen::Matrix3f sigma;
    solveLine(instance, mu, normal, lambda, sigma);

    pole_cloud_ += instance;

    if(lambda(0) > 0.3 * lambda(2)) {
      continue;
    }

    Eigen::Vector3d point_a = (mu + normal * std::sqrt(lambda(2) * 2)).cast<double>();
    Eigen::Vector3d point_b = (mu - normal * std::sqrt(lambda(2) * 2)).cast<double>();
    double length = (point_b - point_a).norm();
    if (length < 1.0)
      continue;

    pcl::SampleConsensusModelLine<pcl::PointXYZI>::Ptr model_line(
        new pcl::SampleConsensusModelLine<pcl::PointXYZI>(instance.makeShared())
    );
    pcl::RandomSampleConsensus<pcl::PointXYZI> ransac(model_line);
    ransac.setDistanceThreshold(0.15);
    ransac.setMaxIterations(10);
    ransac.computeModel();
    std::vector<int> inliers;
    ransac.getInliers(inliers);
    double inlier_ratio = (double) inliers.size() / instance.size();
    if (inlier_ratio < 0.7 || inliers.size() < 20) {
      continue;
    }

    pcl::PointCloud<pcl::PointXYZI> inlier_instance;
    pcl::copyPointCloud<pcl::PointXYZI>(instance, inliers, inlier_instance);

    int intensity = rand() % 10000;
    for(auto &point: inlier_instance.points) {
      point.intensity = intensity;
    }
    line_clusters_.push_back(inlier_instance.makeShared());
    // pole_cloud_ += inlier_instance;
    valid_cluster_num++;
  }

  // pole_cloud_ = pole_cloud;
}

void VLPExtractor::extractSurface(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, int scan_line) {
  double tg, tTgt;
  pcl::PointCloud<pcl::PointXYZI> road_cloud;
  pcl::PointCloud<pcl::PointXYZI> struct_cloud;
  travel::estimateGround(*(cloud), road_cloud, struct_cloud, tg);

  travel::ObjectCluster<pcl::PointXYZI> travel_object_seg(1.0, 150.0, -25.0, 25.0, scan_line);
  std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> cluster_cloud;
  travel_object_seg.segmentObjects(struct_cloud.makeShared(), cluster_cloud);

  refineSurface(road_cloud.makeShared(), road_clusters_, ROAD_ID);
  for(int i = 0; i < cluster_cloud.size(); ++i) {
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> struct_clusters;
    refineSurface(cluster_cloud[i], struct_clusters, BUILDING_ID);
    struct_clusters_.insert(struct_clusters_.end(), struct_clusters.begin(), struct_clusters.end());
  }

}

void VLPExtractor::refineSurface(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud, std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr>& clusters, const uint16_t sem_type) {

  std::unordered_map<VoxelLoc, OctoVoxel::Ptr> ovmap;

  Transform Twl;
  pcl::VoxelGrid<pcl::PointXYZI> vf;
  vf.setLeafSize(0.2, 0.2, 0.2);
  vf.setInputCloud(cloud);
  vf.filter(*cloud);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_tmp(new pcl::PointCloud<pcl::PointXYZI>());
  for(auto point: cloud->points) {
    pcl::PointXYZI pt;
    pt.x = point.x;
    pt.y = point.y;
    pt.z = point.z;
    pt.intensity = point.intensity;
    cloud_tmp->push_back(pt);
  }

  // allocate hash octree and cut the cloud
  const double max_res = (sem_type == ROAD_ID) ? 8.0 : 16.0;
  OctoVoxel::max_level_ = (sem_type == ROAD_ID) ? 4 : 5;
  const double thres = (sem_type == ROAD_ID) ? 0.02 : 0.075;
  // const double thres = (sem_type == ROAD_ID) ? 0.03 : 0.05;
  cutCloud(*cloud_tmp, Twb_, sem_type, Eigen::Vector3f(max_res, max_res, max_res), ovmap);

  srand((unsigned)time(NULL));
  std::vector<pcl::PointCloud<pcl::PointXYZI>> leaf_nodes;
  std::vector<OctoVoxel> surf_voxels;
  for(auto iter = ovmap.begin(); iter != ovmap.end(); ++iter) {
    iter->second->parse(thres);
    std::vector<OctoVoxel> vec_voxel;
    iter->second->getLeafVoxel(vec_voxel);
    surf_voxels.insert(surf_voxels.end(), vec_voxel.begin(), vec_voxel.end());
  }

  assert(leaf_nodes.size() == surf_voxels.size());
  clusters.clear();
  for(auto node: surf_voxels) {
    int intensity = rand() % 10000;
    for(auto &point: node.cloud_) {
      point.getVector3fMap() = point.getVector3fMap();
      point.intensity = intensity;
    }
    clusters.push_back(node.cloud_.makeShared());
  }
}

void VLPExtractor::constructOb() {
  for(auto line_cloud_ptr: line_clusters_) {
    Eigen::Vector3f mu, normal, lambda;
    Eigen::Matrix3f sigma;
    solveLine(*line_cloud_ptr, mu, normal, lambda, sigma);
    double angle = std::acos(normal.dot(Eigen::Vector3f(0, 0, 1))) * 180 / M_PI;
    if(angle > 20 && angle < 160)
      continue;

    LineOB::Ptr ob(new LineOB(POLE_ID, line_cloud_ptr));
    vec_line_feature_.push_back(ob);
    if(ob->point_num_ == 0) {
      printf("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
  }

  for(auto road_cloud_ptr: road_clusters_) {
    Eigen::Matrix4f Tbw = Twb_.inverse().matrix().cast<float>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_road_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    local_road_cloud_ptr->resize(road_cloud_ptr->size());
    // pcl::transformPointCloud(*road_cloud_ptr, *local_road_cloud_ptr, Tbw);
    for(int i = 0; i < road_cloud_ptr->size(); i++) {
      local_road_cloud_ptr->points[i].getVector3fMap() = Tbw.block<3, 3>(0, 0) * road_cloud_ptr->points[i].getVector3fMap() + Tbw.block<3, 1>(0, 3);
    }
    SurfaceOB::Ptr ob(new SurfaceOB(ROAD_ID, local_road_cloud_ptr));
    vec_surf_feature_.push_back(ob);
    if(ob->point_num_ == 0) {
      printf("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
  }

  for(auto struct_cloud_ptr: struct_clusters_) {
    Eigen::Matrix4f Tbw = Twb_.inverse().matrix().cast<float>();
    pcl::PointCloud<pcl::PointXYZI>::Ptr local_struct_cloud_ptr(new pcl::PointCloud<pcl::PointXYZI>());
    // pcl::transformPointCloud(*road_cloud_ptr, *local_road_cloud_ptr, Tbw);
    local_struct_cloud_ptr->resize(struct_cloud_ptr->size());
    for(int i = 0; i < struct_cloud_ptr->size(); i++) {
      local_struct_cloud_ptr->points[i].getVector3fMap() = Tbw.block<3, 3>(0, 0) * struct_cloud_ptr->points[i].getVector3fMap() + Tbw.block<3, 1>(0, 3);
    }
    SurfaceOB::Ptr ob(new SurfaceOB(BUILDING_ID, local_struct_cloud_ptr));
    if(ob->point_num_ == 0) {
      printf("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
    vec_surf_feature_.push_back(ob);
  }

}

pcl::PointCloud<pcl::PointXYZI>::Ptr VLPExtractor::getEdgeCloud() {
  // return pole_cloud_.makeShared();
  pcl::PointCloud<pcl::PointXYZI>::Ptr edge_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  for(int row = 0; row < h_; ++row) {
    for(int col = 0; col < w_; ++col) {
      float grad = img_[row][col].grad;
      // if(grad > 0.3f) {
      if(img_[row][col].attribute == PointAttribute::POLE) {
        pcl::PointXYZI point;
        point.getVector3fMap() = img_[row][col].pos;
        point.intensity = 50;
        edge_cloud->push_back(point);
      }
      if(img_[row][col].attribute == PointAttribute::EDGE_LEFT) {
        pcl::PointXYZI point;
        point.getVector3fMap() = img_[row][col].pos;
        point.intensity = 100;
        edge_cloud->push_back(point);
      }
      else if(img_[row][col].attribute == PointAttribute::EDGE_RIGHT) {
        pcl::PointXYZI point;
        point.getVector3fMap() = img_[row][col].pos;
        point.intensity = 150;
        edge_cloud->push_back(point);
      }
    }
  }
  return edge_cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr VLPExtractor::getSurfCloud() {
  return surf_cloud_.makeShared();
}

pcl::PointCloud<pcl::PointXYZI>::Ptr VLPExtractor::getRawSurfCloud() {
  return raw_surf_cloud_.makeShared();
}

} // namespace SLIM
