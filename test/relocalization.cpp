#include <iostream>
#include <chrono>
#include <thread>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>  
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>

#include "frame.h"
#include "viewer.h"
#include "vector_map.h"
#include "observation.h"
#include "pcm_solver.h"
#include "locator.h"
#include "extractor.h"

struct KfInfo {
  std::string filename;
  SLIM::Transform Twb;
};


std::vector<std::string> readTxtFile(const std::string& dirname) {
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

    size_t pos = filename.find('.');
    if (pos != std::string::npos) {
      std::string sub_str = filename.substr(pos + 1);
      if(sub_str == "txt") {
        filenames.push_back(dirname + "/" + filename);
      }
    } else {
      std::cout << "No '.' found in the string." << std::endl;
      continue;
    }
  }
  closedir(dir);

  std::sort(filenames.begin(), filenames.end());
  return filenames;
}

std::vector<std::string> readJsonFile(const std::string& dirname) {
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

    size_t pos = filename.find('.');
    if (pos != std::string::npos) {
      std::string sub_str = filename.substr(pos + 1);
      if(sub_str == "json") {
        filenames.push_back(dirname + "/" + filename);
      }
    } else {
      std::cout << "No '.' found in the string." << std::endl;
      continue;
    }
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
    if(!(iss >> name)) {
      break;
    }
      
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



int main(int argc, char **argv) {
  if(argc < 2) {
    std::cerr << "Usage: <main> kitti_root_path ... " << std::endl;
    return -1;
  }
  cv::Mat bottom(100, 100, CV_8U, cv::Scalar(255));
  std::string map_path = argv[1];
  std::string traj_path = argv[2];
  std::string output_path = argv[3];
  createDirectory(output_path);

  Viewer::Ptr viewer = Viewer::Ptr(new Viewer());
  std::thread vis_thread = std::thread(&Viewer::Run, viewer);

  std::vector<std::string> map_files = readJsonFile(map_path);
  std::sort(map_files.begin(), map_files.end());
  std::string map_file = map_files.back();

  std::vector<std::string> traj_files = readTxtFile(traj_path);
  std::vector<std::map<uint64_t, KfInfo>> trajs;
  for(int i = 0; i < traj_files.size(); i++) {
    auto traj = readTrajectory(traj_files[i]);
    trajs.push_back(traj);
  }
  
  VectorMap::Ptr map = VectorMap::Ptr(new VectorMap);
  map->loadJsonFile(map_file);

  viewer->SetRefMap(map);

  std::map<uint64_t, SLIM::Transform> map_kfs;
  for(auto kf_iter: map->getKeyFrames()) {
    Frame::Ptr kf = kf_iter.second;
    map_kfs.insert(std::make_pair(kf->timestamp(), kf->Twb()));
  }

  for(int i = 0; i < trajs.size(); i++) {
    // uint64_t start_timestamp;
    // SLIM::Transform start_Tf;

    std::map<uint64_t, KfInfo>::iterator start_iter;
    SLIM::Locator locator;
    SLIM::Transform map_Tf;
    locator.setMap(map);
    locator.setViewer(viewer);
    std::string output_file_name = output_path + "/locator_traj_" + std::to_string(i) + ".txt";
    locator.setTrajFileName(output_file_name);

    for(auto iter: trajs[i]) {
      uint64_t timestamp = iter.first;
      if(map_kfs.find(timestamp) != map_kfs.end()) {
        // start_timestamp = timestamp;
        map_Tf = map_kfs[timestamp];
        start_iter = trajs[i].find(timestamp);
        break;
      }
    } 
    // printf("Found First Frame \n");
    // std::cout << "start pose: " << std::endl << map_Tf.matrix() << std::endl;
    int kf_index = 0;
    for(; std::next(start_iter) != trajs[i].end(); start_iter++) {
      kf_index++;
      printf("Process kf: %d, seq: %d\n", kf_index, i);
      uint64_t cur_timestamp = start_iter->first;
      SLIM::Transform cur_Tf = start_iter->second.Twb;
      if(!locator.initialized_) {
        printf("Start initialization!\n");
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        std::string filename = trajs[i][cur_timestamp].filename;
        if(pcl::io::loadPCDFile(filename, *cloud) == -1) {
          throw std::runtime_error("Read PCD Failed!\n");
        }

        SLIM::VLPExtractor extractor;
        extractor.Twb_ = SLIM::Transform();
        extractor.extractPole(cloud, 128);
        extractor.extractSurface(cloud, 128);

        TicToc timer;
        extractor.constructOb();

        pcl::PointCloud<pcl::PointXYZI> pole, road, building;
        for(auto node: extractor.road_clusters_) {
          road += *node;
        }
        for(auto node: extractor.struct_clusters_) {
          building += *node;
        }
        for(auto node: extractor.line_clusters_) {
          pole += *node;
        }

        pcl::VoxelGrid<pcl::PointXYZI> vf;
        vf.setLeafSize(0.5, 0.5, 0.5);
        vf.setInputCloud(pole.makeShared());
        vf.filter(pole);
        vf.setInputCloud(road.makeShared());
        vf.filter(road);
        vf.setInputCloud(building.makeShared());
        vf.filter(building);

        printf("Preprocess Time Cost: %lf", timer.toc());

        locator.cur_Twb_ = map_Tf;
        locator.last_Twb_ = map_Tf;
        locator.delta_Twb_ = SLIM::Transform();
        if(locator.solve(cur_timestamp, pole.makeShared(), road.makeShared(), building.makeShared())) {
          locator.initialized_ = true;
          printf("Initialize Succeessfully!\n");
        }
        else {
          printf("Initialize Failed!\n");
          throw std::runtime_error("Failed initialization!\n");
          break;
        }

        cv::Mat bottom(100, 100, CV_8U, cv::Scalar(255));
        cv::imshow("bottom", bottom);
        cv::waitKey(0);

      }   
      else {

        TicToc timer;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>());
        std::string filename = trajs[i][cur_timestamp].filename;
        if(pcl::io::loadPCDFile(filename, *cloud) == -1) {
          throw std::runtime_error("Read PCD Failed!\n");
        }

        printf("Loading Time Cost: %lf\n", timer.toc());
        timer.tic();

        SLIM::VLPExtractor extractor;
        extractor.Twb_ = SLIM::Transform();
        extractor.extractPole(cloud, 128);
        extractor.extractSurface(cloud, 128);
        extractor.constructOb();

        pcl::PointCloud<pcl::PointXYZI> pole, road, building;
        for(auto node: extractor.road_clusters_) {
          road += *node;
        }
        for(auto node: extractor.struct_clusters_) {
          building += *node;
        }
        for(auto node: extractor.line_clusters_) {
          pole += *node;
        }


        pcl::VoxelGrid<pcl::PointXYZI> vf;
        vf.setLeafSize(0.5, 0.5, 0.5);
        vf.setInputCloud(pole.makeShared());
        vf.filter(pole);
        vf.setInputCloud(road.makeShared());
        vf.filter(road);
        vf.setInputCloud(building.makeShared());
        vf.filter(building);

        printf("Feature Extraction Time Cost: %lf\n", timer.toc());

        if(locator.solve(cur_timestamp, pole.makeShared(), road.makeShared(), building.makeShared())) {
          printf("[sensorCallback] Localization Warning");
        }
      }   
      // cv::imshow("bottom", bottom);
      // cv::waitKey(0);

    }

    locator.saveTumTrajectory(output_file_name);
    break;
  }

  vis_thread.join();
  return 0;
}