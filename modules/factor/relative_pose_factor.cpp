#include "factor/relative_pose_factor.h"

namespace SLIM {
bool RelativePoseFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Matrix3d Ri{Qi.toRotationMatrix()};

  Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
  Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);
  Eigen::Matrix3d Rj{Qj.toRotationMatrix()};

  Eigen::Vector3d Pij = Ri.transpose() * (Pj - Pi);
  Eigen::Quaterniond Qij = Qi.inverse() * Qj;

  Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
  residual.segment<3>(0) = Pij - rel_pose_.p();
  residual.segment<3>(3) = 2 * (rel_pose_.q().inverse() * Qij).vec();
  residual = sqrt_info_ * residual;

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> dr_dTi(jacobians[0]);
      dr_dTi.setZero();
      Eigen::Matrix3d dt_dti = -Ri.transpose();
      Eigen::Matrix3d dt_dqi = skewSymmetric(Pij);
      Eigen::Matrix3d dq_dqi = -(Qright(Qij) * Qleft(rel_pose_.q().inverse())).bottomRightCorner<3, 3>();
      dr_dTi.block<3, 3>(0, 0) = dt_dti;
      dr_dTi.block<3, 3>(0, 3) = dt_dqi;
      dr_dTi.block<3, 3>(3, 3) = dq_dqi;
      dr_dTi = sqrt_info_ * dr_dTi;
    }
    if(jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> dr_dTj(jacobians[1]);
      dr_dTj.setZero();
      Eigen::Matrix3d dr_dtj = Ri.transpose();
      Eigen::Matrix3d dr_dqj = Qleft(rel_pose_.q().inverse() * Qij).bottomRightCorner<3, 3>();
      dr_dTj.block<3, 3>(0, 0) = dr_dtj;
      dr_dTj.block<3, 3>(3, 3) = dr_dqj;
      dr_dTj = sqrt_info_ * dr_dTj;
    }
  }
  return true;
}
}