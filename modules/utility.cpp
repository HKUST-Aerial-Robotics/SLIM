#include "utility.h"

namespace SLIM {


bool directoryExists(const std::string& path) {
  std::ifstream dir(path);
  return dir.good();
}

bool createDirectory(const std::string& path) {

  if (directoryExists(path)) {
    std::cout << path + " exists!" << std::endl;
    return true;
  }
  std::cout << "Create Path: " << path << std::endl;
  
  if (mkdir(path.c_str(), 0777) == 0) {
    return true;
  }
  return false;
}


void removeFilesInFolder(const std::string& folderPath) {
  DIR* dir = opendir(folderPath.c_str());
  if (dir == nullptr) {
    std::cout << "Failed to open directory." << std::endl;
    return;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    if (strcmp(entry->d_name, ".") != 0 && strcmp(entry->d_name, "..") != 0) {
      std::string filePath = folderPath + "/" + entry->d_name;
      if (remove(filePath.c_str()) != 0) {
        std::cout << "Failed to remove file: " << filePath << std::endl;
      }
    }
  }

  closedir(dir);
}

} // namespace SLIM
