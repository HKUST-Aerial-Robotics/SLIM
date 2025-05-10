#ifndef GRAFF_COORDINATE_H
#define GRAFF_COORDINATE_H

#include <Eigen/Core>

namespace SLIM {
class Graff {
 public:
  Graff() = default;

  Graff(const Eigen::MatrixXd& A, const Eigen::Vector3d& b)
  : A_(A), b_(b) {}

  virtual ~Graff() = default;

  Eigen::MatrixXd A_;
  Eigen::Vector3d b_;
};

class GraffLine {
 public:
  GraffLine(const Eigen::Vector3d& A, const Eigen::Vector3d& b) {
    coord_.resize(4, 2);
    coord_.setZero();
    coord_.block<3, 1>(0, 0) = A;
    coord_.block<3, 1>(0, 1) = b;
    coord_(3, 1) = 1.0;
    // coord_.rightCols<1>() /= std::sqrt(1 + b.squaredNorm());
  }

  ~GraffLine() = default;

  Eigen::MatrixXd get() const {
    return coord_;
  }
 private:
  Eigen::MatrixXd coord_;
};

class GraffSurface {
 public:
  GraffSurface(const Eigen::Matrix<double, 3, 2>& A, const Eigen::Vector3d& b) {
    coord_.resize(4, 3);
    coord_.setZero();
    coord_.block<3, 2>(0, 0) = A;
    coord_.block<3, 1>(0, 2) = b;
    coord_(3, 2) = 1.0;
    // coord_.rightCols<1>() /= std::sqrt(1 + b.squaredNorm());
  }

  ~GraffSurface() = default;

  Eigen::MatrixXd get() const {
    return coord_;
  }

 private:
  Eigen::MatrixXd coord_;
};
}

#endif