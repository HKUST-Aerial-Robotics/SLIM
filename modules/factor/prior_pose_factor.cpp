#include "factor/prior_pose_factor.h"

namespace SLIM {
bool PriorPoseFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
  residual.segment<3>(0) = Pi - prior_pose_.p();
  residual.segment<3>(3) = 2 * (prior_pose_.q().inverse() * Qi).vec();
  residual = sqrt_info_ * residual;

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 7, Eigen::RowMajor>> dr_dTi(jacobians[0]);
      dr_dTi.setZero();
      dr_dTi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      dr_dTi.block<3, 3>(3, 3) = Qleft(prior_pose_.q().inverse() * Qi).bottomRightCorner<3, 3>();
      dr_dTi = sqrt_info_ * dr_dTi;
    }
  }
  return true;
}
}