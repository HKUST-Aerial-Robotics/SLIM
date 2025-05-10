#include "extractor.h"
#include "semantic_definition.h"
#include "vector_map.h"
#include "viewer.h"
#include "multi_map_merger.h"
#include "pcm_solver.h"
#include "utility.h"
#include "clipper.h"
#include <dirent.h>

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
    std::string extension;
    size_t dotPos = filename.find_last_of(".");
    if (dotPos != std::string::npos && dotPos < filename.length() - 1) {
      extension = filename.substr(dotPos + 1);
    }
    else {
      continue;
    }
    if(extension != "json")
      continue;

    filenames.push_back(dirname + "/" + filename);
  }
  closedir(dir);

  std::sort(filenames.begin(), filenames.end());
  return filenames;
}

int main(int argc, char** argv) {

  bool use_ba = true;
  std::string cloud_path, map_save_path, output_path;
  map_save_path = argv[1];
  output_path = argv[2];

  createDirectory(output_path);

  Viewer::Ptr viewer = Viewer::Ptr(new Viewer);
  std::thread vis_thread = std::thread(&Viewer::Run, viewer);

  std::vector<std::string> files = readPath(map_save_path);
  SLIM::MultiMapMerger merger;
  merger.setOutputPath(output_path);
  merger.setViewer(viewer);
  std::vector<VectorMap::Ptr> submaps(files.size());
  for(int i = 0; i < files.size(); ++i) {
    submaps[i] = VectorMap::Ptr(new VectorMap);
  }

#pragma omp parallel for shared(submaps, files) 
  for(int i = 0; i < files.size(); ++i) {
    submaps[i]->loadJsonFile(files[i]);
  }
  merger.ref_map_ = submaps[0];
  SLIM::Transform T_os = submaps[0]->getKeyFrames()[0]->Twb();
  Eigen::Vector3d p = T_os.p() + Eigen::Vector3d(40.0, 40.0, 40.0);
  T_os = SLIM::Transform(p, T_os.dcm());
  merger.ref_map_->transform(T_os.inverse());
    
  // submaps[0]->extractSparseStruct();
  std::vector<VectorMap::Ptr> cache_map;
  std::vector<int> cache_id;
  std::vector<int> order;


  cv::Mat bottom(100, 100, CV_8U, cv::Scalar(255));
  viewer->SetRefMap(submaps[0]);
  cv::imshow("bottom", bottom);
  cv::waitKey(0);

  for(int i = 1; i < submaps.size(); i++) {
    merger.cur_map_ = submaps[i];
    T_os = submaps[i]->getKeyFrames()[0]->Twb();
    merger.cur_map_->transform(T_os.inverse());
    bool res = merger.merge();
    if(!res) {
      cache_map.push_back(submaps[i]);
      cache_id.push_back(i);
    }
    else {
      order.push_back(i);
    }
  }

  for(int i = 0; i < cache_map.size(); i++) {
    merger.cur_map_ = cache_map[i];
    bool res = merger.merge();
    if(res) {
      order.push_back(cache_id[i]);
    }
  }
  std::cout << "cache size: " << cache_map.size() << std::endl;
  std::cout << "merge order: " << std::endl;
  for(auto num: order) {
    std::cout << num << std::endl;
  }
  vis_thread.join();
  return 0;
}
