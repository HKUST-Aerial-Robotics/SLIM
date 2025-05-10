#ifndef UTILITY_H
#define UTILITY_H

#include <ctime>
#include <cstdlib>
#include <chrono>

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <ctime>
#include <dirent.h>
#include "semantic_definition.h"

#include <opencv4/opencv2/opencv.hpp>

#include <Eigen/Core>

namespace SLIM {
  
class TicToc {
 public:
  TicToc() {
    tic();
  }

  void tic() {
    start = std::chrono::system_clock::now();
  }

  double toc() {
    end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    return elapsed_seconds.count() * 1000;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start, end;
};

template <typename T>
void reduceVector(std::vector<T> &v, const std::vector<int>& set) {
  int j = 0;
  for (int i = 0; i < int(set.size()); i++) {
    v[j++] = v[set[i]];
  }
  v.resize(j);
}

class MatrixVisualizer {
 public:
  MatrixVisualizer() = default;
  ~MatrixVisualizer() = default;

  MatrixVisualizer(const uint32_t row, const uint32_t col) {
    scale_factor_ = (float)max_dim_ / (float)std::max(row, col);
    if(scale_factor_ >= 1) {
      uint32_t scale = (uint32_t)std::round(scale_factor_);
      vis_mat_ = cv::Mat(row * scale, col * scale, CV_8UC3, cv::Scalar(0, 0, 0));
    }
  }

  void render(const Eigen::MatrixXd& mat, const std::string& title) {
    uint32_t row = mat.rows(), col = mat.cols();
    double normalize_coeff = std::max(std::abs(mat.maxCoeff()), std::abs(mat.minCoeff()));
    vis_mat_ = cv::Mat(row, col, CV_8UC3, cv::Scalar(255, 255, 255));
    for(int i = 0; i < row; ++i) {
      for(int j = 0; j < col; ++j) {
        if(mat(i, j) > 0) {
          double scale = std::max(std::min(1.0, mat(i, j) / max_element_), 0.4);
          auto color = cv::Vec3b((uchar)std::round(255 - (255 - CV_COLOR_DEEPPINK[0]) * scale), 
                            (uchar)std::round(255 - (255 - CV_COLOR_DEEPPINK[1]) * scale), 
                            (uchar)std::round(255 - (255 - CV_COLOR_DEEPPINK[2]) * scale));
          // auto color = cv::Vec3b((uchar)std::round(255 * (1-scale) - CV_COLOR_DEEPPINK[0] * scale), 
          //                   (uchar)std::round(255 * (1-scale) - CV_COLOR_DEEPPINK[1] * scale), 
          //                   (uchar)std::round(255 * (1-scale) - CV_COLOR_DEEPPINK[2] * scale));
          vis_mat_.at<cv::Vec3b>(i, j) = color;     
          // vis_mat_.at<cv::Vec3b>(i, j) = cv::Vec3b(CV_COLOR_DEEPPINK[0], CV_COLOR_DEEPPINK[1], CV_COLOR_DEEPPINK[2]);
        }
        else if(mat(i, j) < 0) {
          double scale = -std::min(-0.4, std::max(-1.0, mat(i, j) / max_element_));

          // auto color = cv::Vec3b((uchar)std::round(255 * (1-scale) - CV_COLOR_DARKCYAN[0] * scale), 
          //                   (uchar)std::round(255 * (1-scale) - CV_COLOR_DARKCYAN[1] * scale), 
          //                   (uchar)std::round(255 * (1-scale) - CV_COLOR_DARKCYAN[2] * scale));
          auto color = cv::Vec3b((uchar)std::round(255 - (255 - CV_COLOR_DARKCYAN[0]) * scale), 
                            (uchar)std::round(255 - (255 - CV_COLOR_DARKCYAN[1]) * scale), 
                            (uchar)std::round(255 - (255 - CV_COLOR_DARKCYAN[2]) * scale));
          vis_mat_.at<cv::Vec3b>(i, j) = color;
          // vis_mat_.at<cv::Vec3b>(i, j) = cv::Vec3b(CV_COLOR_DARKCYAN[0], CV_COLOR_DARKCYAN[1], CV_COLOR_DARKCYAN[2]);
        }
        else {
          vis_mat_.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
        }
      }
    }
    cv::imshow("vis", vis_mat_);
    cv::waitKey(0);
    if(title.length() > 0)
      cv::imwrite(title, vis_mat_);
  }

  float scale_factor_;
  const uint32_t max_dim_ = 640;
  const double max_element_ = 10;
  cv::Mat vis_mat_;
};

bool createDirectory(const std::string& path);

bool directoryExists(const std::string& path);

void removeFilesInFolder(const std::string& folderPath);

} // namespace SLIM





#endif