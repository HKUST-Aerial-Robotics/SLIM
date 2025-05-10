
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <thread>
#include <time.h>

#include "extractor.h"
#include "semantic_definition.h"
#include "vector_map.h"
#include "viewer.h"
#include "utility.h"
#include <sys/stat.h>

std::string map_save_path, seq, odom_save_path, gt_traj_save_path, cloud_path, pose_file;
double start, duration;

VectorMap::Ptr vector_map;
Viewer::Ptr viewer;
std::thread visualization;
std::string dataset_name = "helipr";

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

void formatCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, int line_num) {
  int prev_quadrant = 0;
  int ring = 0;
  std::vector<std::vector<ScanPoint>> scan_img;
  scan_img.resize(line_num);
  std::cout << "before preprocess: " << cloud->size() << std::endl;
  for(int i = 0; i < cloud->size(); ++i) {
    pcl::PointXYZI pt;
    pt = cloud->points[i];
    int quadrant = getQuadrant(pt);
    std::cout << "point index: " << i << " quadrant: " << quadrant << " pitch: " << std::atan2(pt.z, std::sqrt(pt.x*pt.x + pt.y*pt.y)) * 180 / M_PI << std::endl;
    if((quadrant == 1) && (prev_quadrant == 4)) {
      ring ++;
    }
    if(ring == line_num)
      break;

    Eigen::Vector3f pos = pt.getVector3fMap();
    float yaw = std::atan2(pos.y(), pos.x());
    scan_img[ring].push_back(ScanPoint(pos, yaw));
    prev_quadrant = quadrant;
  }
  std::cout << "preprocess line: " << scan_img.size() << std::endl;
  for(int line_id = 0; line_id < line_num; ++line_id) {
    std::sort(scan_img[line_id].begin(), scan_img[line_id].end(), [&](const ScanPoint& p1, const ScanPoint& p2) {
      return p1.yaw > p2.yaw;
    });
  }
  cloud->clear();
  for(int i = 0; i < line_num; ++i) {
    for(int j = 0; j < scan_img[i].size(); ++j) {
      pcl::PointXYZI pt;
      pt.getVector3fMap() = scan_img[i][j].pos;
      pt.intensity = i;
      cloud->push_back(pt);
    }
  }
  std::cout << "after preprocess: " << cloud->size() << std::endl;
}


std::map<uint64_t, KfInfo> readTrajectory(const std::string& file_name) {
  std::map<uint64_t, KfInfo> trajectory;
  std::ifstream fin(file_name);
  std::string line;
  while (std::getline(fin, line)) {
    std::istringstream iss(line);
    uint64_t timestamp;
    printf("here\n");
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
    printf("%s\n", name.c_str());
    trajectory.insert(std::make_pair(timestamp, info));
  }
  fin.close();
  return trajectory;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr readKittiBin(const std::string& file_name) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  std::ifstream fcloud(file_name.c_str(), std::ios::binary);
  if (!fcloud) {
    std::cerr << "[Error][readKittiBin] Point Cloud File '" << file_name << "' is not found!" << std::endl;
    return 0;
  }

  // get length of input file:
  fcloud.seekg(0, fcloud.end);
  int cloud_length = (int)fcloud.tellg();
  fcloud.seekg(0, fcloud.beg);

  int cloud_size = cloud_length / sizeof(float);
  cloud->reserve(cloud_size);

  std::vector<float> point_buffer(cloud_size);
  fcloud.read(reinterpret_cast<char*>(&point_buffer[0]), cloud_size*sizeof(float));
  fcloud.close();

  pcl::PointXYZI point;
  for (uint32_t i = 0; i < point_buffer.size(); i+=4) {
    point.x = point_buffer[i];
    point.y = point_buffer[i+1];
    point.z = point_buffer[i+2];
    point.intensity = 0;
    cloud->push_back(point);
  }
  formatCloud(cloud, 64);
  return cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr readM2DGRPcd(const std::string& file_name) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
  int scan_line = 32;
  if(pcl::io::loadPCDFile(file_name, *cloud) == -1) {
    throw std::runtime_error("Read PCD Failed!\n");
  }     
  return cloud;
}


VectorMap::Ptr buildMap(const std::map<uint64_t, KfInfo>& traj) {
  SLIM::Transform last_Twb;
  bool init_flag = false;
  VectorMap::Ptr map(new VectorMap);

  // pcl::visualization::PCLVisualizer viewer("viewer");
  for(auto iter: traj) {
    uint64_t timestamp = iter.first;
    SLIM::Transform Twb = iter.second.Twb;
    std::string filename = iter.second.filename;

    if(init_flag) {
      SLIM::Transform delta_Twb = last_Twb.inverse() * Twb;
      double dist = delta_Twb.p().norm();
      if(dist < 2)
        continue;
    }

    std::vector<std::vector<ScanPoint>> range_img;
    int scan_line = 128;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
    if(dataset_name == "helipr") {
      scan_line = 128;
      if(pcl::io::loadPCDFile(filename, *cloud) == -1) {
        throw std::runtime_error("Read PCD Failed!\n");
      }
    }
    else if(dataset_name == "m2dgr") {
      scan_line = 32;
      cloud = readM2DGRPcd(filename);
      printf("Read M2DGR PCD Done!\n");
    }
    else if(dataset_name == "kitti") {
      scan_line = 64;
      cloud = readKittiBin(filename);
    }
    else if(dataset_name == "nclt") {
      scan_line = 32;
      // cloud = readNcltBin(filename);
      if(pcl::io::loadPCDFile(filename, *cloud) == -1) {
        throw std::runtime_error("Read PCD Failed!\n");
      }
      // preprocess(cloud, scan_line);
    }

    Frame::Ptr frame = Frame::Ptr(new Frame(timestamp, Twb));
    SLIM::VLPExtractor extractor;
    extractor.Twb_ = Twb;
    extractor.extractPole(cloud, scan_line);
    extractor.extractSurface(cloud, scan_line);
    extractor.constructOb();

    frame->line_obs_ = extractor.vec_line_feature_;
    frame->surface_obs_ = extractor.vec_surf_feature_;
    frame->cloud_path_ = filename;

    // std::cout << "line: " << frame->line_obs_.size() << std::endl;
    // std::cout << "surf: " << frame->surface_obs_.size() << std::endl;
    
    Eigen::Matrix<double, 6, 6> sqrt_info = Eigen::Matrix<double, 6, 6>::Identity();
    sqrt_info.block<3, 3>(0, 0) *= 1/0.1;
    sqrt_info.block<3, 3>(3, 3) *= 1/0.01;
    map->addOdomInfo(frame, sqrt_info);
    map->alignFrame(frame);
    map->resolveLM();
    std::cout << "Frame Registration Done!" << std::endl;

    init_flag = true;
    last_Twb = Twb;
  }   

  return map;

}

int main(int argc, char** argv) {

  std::string traj_path = argv[1];
  std::string map_path = argv[2];
  dataset_name = argv[3];
  std::vector<std::string> traj_names = readPath(traj_path);
  std::cout << "Trajectory Path: " << traj_path << std::endl;

  for(int traj_id = 0; traj_id < traj_names.size(); traj_id++) {
    if(!directoryExists(map_path)) {
      createDirectory(map_path);
    }
    std::cout << "Map Path: " << map_path << std::endl;

    auto traj_name = traj_names[traj_id];
    std::cout << "Traj Path: " << traj_name << std::endl;
    auto traj = readTrajectory(traj_name);

    VectorMap::Ptr map = buildMap(traj);
    map->prune(1.414);
  
    map->saveJsonFile(map_path + "/submap_" + std::to_string(traj_id) + ".json");
    map->releaseAll();
  }
  return 0;
}