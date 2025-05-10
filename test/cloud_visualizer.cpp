
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <thread>
#include <time.h>

#include "extractor.h"
#include "semantic_definition.h"
#include "vector_map.h"
#include "viewer.h"
#include "factor/line_info.h"
#include "factor/surface_info.h"
#include <sys/stat.h>

// #include <pcl/geometry/create_elliptical_mesh.h>.

std::string map_save_path, seq, odom_save_path, gt_traj_save_path, cloud_path, pose_file;
double start, duration;

VectorMap::Ptr vector_map;
Viewer::Ptr viewer;
std::thread visualization;

// bool init = false;
// SLIM::Transform last_Twb;

struct KfInfo {
  std::string filename;
  SLIM::Transform Twb;
};

std::vector<std::string> readPath(const std::string& dirname) {
  std::vector<std::string> filenames;
  // sweep the directory
  DIR* dir;
  if ((dir = opendir(dirname.c_str())) == nullptr) {
    throw std::runtime_error("directory " + dirname + " does not exist");
  }
  dirent* dp;
  for (dp = readdir(dir); dp != nullptr; dp = readdir(dir)) {
    const std::string filename = dp->d_name;
    if (filename == "." || filename == "..") {
      continue;
    }
    filenames.push_back(dirname + "/" + filename);
  }
  closedir(dir);

  std::sort(filenames.begin(), filenames.end());
  return filenames;
}

std::map<uint64_t, KfInfo> readTrajectory(const std::string& file_name) {
  std::map<uint64_t, KfInfo> trajectory;
  std::ifstream fin(file_name);
  std::string line;
  while (std::getline(fin, line)) {
    std::istringstream iss(line);
    uint64_t timestamp;
    if (!(iss >> timestamp)) {
      break;
    }
    double data[7];
    if (!(iss >> data[0] >> data[1] >> data[2] >> data[3] >> data[4] >> data[5] >> data[6])) {
      break;
    }
    std::string name;
    iss >> name;

    Eigen::Vector3d t(data[0], data[1], data[2]);
    Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
    SLIM::Transform T(t, q);
    
    KfInfo info;
    info.Twb = T;
    info.filename = name;
    trajectory.insert(std::make_pair(timestamp, info));
  }
  fin.close();
  return trajectory;
}


int main(int argc, char** argv) {


  // std::string traj_file = argv[1];
  std::string map_file = argv[1];
  double intensity = std::stod(argv[2]);
  // auto traj = readTrajectory(traj_file);
  VectorMap::Ptr map(new VectorMap());
  map->loadJsonFile(map_file);
  pcl::visualization::PCLVisualizer viewer("viewer");
  viewer.setBackgroundColor(255, 255, 255);

  auto keyframes = map->getKeyFrames();
  for(auto kf_iter: keyframes) {
    SLIM::Transform T = kf_iter.second->Twb();
    Eigen::Vector3f t = T.p().cast<float>();
    Eigen::Matrix3f R = T.dcm().cast<float>();
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    transform.translation() = t;  // 平移向量
    transform.rotate(R);  // 绕Z轴旋转45度

    // 添加坐标系
    viewer.addCoordinateSystem(1.0, transform);
  }


  int id = 0;
  for(auto kf_iter: keyframes) {
    SLIM::Transform T = kf_iter.second->Twb();
    std::string filename = kf_iter.second->cloud_path_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if(pcl::io::loadPCDFile(filename, *cloud) == -1) {
      throw std::runtime_error("Read PCD Failed!\n");
    }

    pcl::VoxelGrid<pcl::PointXYZI> vf;
    vf.setLeafSize(0.2, 0.2, 0.2);
    vf.setInputCloud(cloud);
    vf.filter(*cloud);

    Eigen::Matrix4f Tf = T.matrix().cast<float>();
    for(auto &p: cloud->points) {
      Eigen::Vector3f gp = Tf.block<3, 3>(0, 0) * p.getVector3fMap() + Tf.block<3, 1>(0, 3);
      p.getVector3fMap() = gp;
    }

    std::string label = "cloud_" + std::to_string(id++);
    // 将点云添加到PCLVisualizer中
    viewer.addPointCloud<pcl::PointXYZI>(cloud, label);

    // 设置点云透明度
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, intensity, label);
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 135.0/255.0, 206.0/255.0, 235.0/255.0, label);
  }

  auto line_lms = map->getSemLines();
  auto surf_lms = map->getSemSurfaces();
  for(auto &sem_iter: line_lms) {
    for(auto &lm_iter: sem_iter.second) {
      LineLM::Ptr& lm = lm_iter.second;
      lm->double2vector();
      auto pa = lm->pa(), pb = lm->pb();
      pcl::PointXYZ p0(pa(0), pa(1), pa(2)), p1(pb(0), pb(1), pb(2));
      std::string label = "line_" + std::to_string(lm->id());
      viewer.addLine(p0, p1, 0, 255, 255, label);
      viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, label);
    }
  }

  for(auto &sem_iter: surf_lms) {
    for(auto &lm_iter: sem_iter.second) {
      SurfaceLM::Ptr& lm = lm_iter.second;
      lm->double2vector();
      auto vertices = lm->vertices();

      pcl::PointCloud<pcl::PointXYZ>::Ptr surf(new pcl::PointCloud<pcl::PointXYZ>());
      surf->points.resize(4);
      surf->points[0] = pcl::PointXYZ(vertices[0](0), vertices[0](1), vertices[0](2));
      surf->points[1] = pcl::PointXYZ(vertices[1](0), vertices[1](1), vertices[1](2));
      surf->points[2] = pcl::PointXYZ(vertices[2](0), vertices[2](1), vertices[2](2));
      surf->points[3] = pcl::PointXYZ(vertices[3](0), vertices[3](1), vertices[3](2));

      pcl::Vertices vertices0;
      vertices0.vertices.resize(3);
      vertices0.vertices[0] = 0;
      vertices0.vertices[1] = 1;
      vertices0.vertices[2] = 2;

      pcl::Vertices vertices1;
      vertices1.vertices.resize(3);
      vertices1.vertices[0] = 0;
      vertices1.vertices[1] = 1;
      vertices1.vertices[2] = 3;

      std::vector<pcl::Vertices> all_vertices;
      all_vertices.push_back(vertices0);
      all_vertices.push_back(vertices1);

      std::string label;
      if(lm->semantic_type() == ROAD_ID) {
        label = "road_" + std::to_string(lm->id());
      }
      else {
        label = "building_" + std::to_string(lm->id());
      }
      viewer.addPolygonMesh<pcl::PointXYZ>(surf, all_vertices, label);

      if(lm->semantic_type() == ROAD_ID) {
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255.0/255.0, 0.0/255.0, 255.0/255.0, label);
      }
      else {
        viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 255.0/255.0, 69.0/255.0, 0.0/255.0, label);
      }
      
      // viewer.addLine(p0, p1, 0, 255, 255, label);
      // viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 5, label);
    }
  }

  viewer.addCoordinateSystem();          
  viewer.spin();
  return 0;
}