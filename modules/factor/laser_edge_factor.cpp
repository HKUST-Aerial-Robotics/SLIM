#include "factor/laser_edge_factor.h"

namespace SLIM {
bool LaserPointToLineFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

  Eigen::Matrix3d sqrt_info_mat = Eigen::Matrix3d::Identity() * sqrt_info_; 
  Eigen::Matrix3d dmat =  Eigen::Matrix3d::Identity() - n_ * n_.transpose();
  Eigen::Map<Eigen::Vector3d> residual(residuals);
  residual = sqrt_info_mat * dmat * (Qi * p_ + Pi - q_);

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dgp_dTi;
      dgp_dTi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      dgp_dTi.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(p_);
      drdT.block<3, 6>(0, 0) = sqrt_info_mat * dmat * dgp_dTi;
    }
  }
  return true;
}


bool LaserEdgePriorFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  LineInfo line_info(Eigen::Vector4d(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]));
  // std::cout << "line : " << line_info.parameters().transpose() << std::endl;
  auto normal = line_info.get_normal();
  auto center = line_info.get_center();
  Eigen::Matrix3d dmat = (Eigen::Matrix3d::Identity() - normal * normal.transpose());
  Eigen::Matrix<double, 6, 6> sqrt_info_mat = Eigen::Matrix<double, 6, 6>::Identity() * sqrt_info_;  

  Eigen::Map<Eigen::Matrix<double, 6, 1>> residual(residuals);
  residual.head<3>() = sqrt_info_mat.block<3, 3>(0, 0) * line_info.distance(pa_);
  residual.tail<3>() = sqrt_info_mat.block<3, 3>(3, 3) * line_info.distance(pb_);
  // std::cout << "residual: " << residual.transpose() << std::endl;

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> drdlm(jacobians[0]);
      drdlm.setZero();
      Eigen::Matrix<double, 6, 6> dXdnc;
      Eigen::Vector3d vec_a = pa_ - center;
      Eigen::Vector3d vec_b = pb_ - center;
      dXdnc.block<3, 3>(0, 0) = -(Eigen::Matrix3d::Identity() * (normal.transpose() * vec_a) + vec_a * normal.transpose());
      dXdnc.block<3, 3>(0, 3) = -dmat;
      dXdnc.block<3, 3>(3, 0) = -(Eigen::Matrix3d::Identity() * (normal.transpose() * vec_b) + vec_b * normal.transpose());
      dXdnc.block<3, 3>(3, 3) = -dmat;
      drdlm = sqrt_info_mat * dXdnc * line_info.jacobian();
    }
  }
  // printf("LaserEdgePriorFactor Evaluate Success!\n");
  return true;
}


// residual dim 2
bool LaserEdgeFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Vector3d global_point = Qi * local_point_ + Pi;

  LineInfo line_info(Eigen::Vector4d(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]));
  auto normal = line_info.get_normal();
  auto center = line_info.get_center();
  Eigen::Matrix3d Rzv = line_info.Rvz().transpose();
  Eigen::Matrix3d dmat = (Eigen::Matrix3d::Identity() - normal * normal.transpose());

  Eigen::Vector3d error = line_info.distance(global_point);
  double sinr = std::sin(parameters[1][0]), cosr = std::cos(parameters[1][0]);
  double sinp = std::sin(parameters[1][1]), cosp = std::cos(parameters[1][1]);
  Eigen::Map<Eigen::Vector2d> residual(residuals);
  residual = sqrt_info_ * Rzv.block<2, 3>(0, 0) * error;

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dqdT;
      dqdT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      dqdT.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(local_point_);
      drdT.block<2, 6>(0, 0) = sqrt_info_ * Rzv.block<2, 3>(0, 0) * dmat * dqdT;
    }
    if(jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> drdlm(jacobians[1]);
      drdlm.setZero();
      // Eigen::Matrix<double, 3, 6> dXdnc;
      // Eigen::Vector3d vec = global_point - center;
      // dXdnc.leftCols<3>() = -(Eigen::Matrix3d::Identity() * (normal.transpose() * vec) + vec * normal.transpose());
      // dXdnc.rightCols<3>() = -dmat;

      // {
      //   Eigen::Matrix<double, 2, 2> drdpr;
      //   drdpr(0, 0) = error(1) * sinp * cosr - error(2) * sinp * sinr;
      //   drdpr(1, 0) = -error(1) * sinr  - error(2) * cosr;
      //   drdpr(0, 1) = -error(0) * sinp + error(1) * cosp * sinr + error(2) * cosp * cosr;
      //   drdpr(1, 1) = 0.0;
      //   drdlm = Rzv.block<2, 3>(0, 0) * dXdnc * line_info.jacobian();
      //   drdlm.block<2, 2>(0, 0) += drdpr;
      //   std::cout << "jacobian1" << std::endl;
      //   std::cout << drdlm << std::endl;
      //   // drdlm = sqrt_info_ * drdlm;        
      // }
    
      Eigen::Matrix<double, 2, 4> drdpr;
      drdpr.setZero();
      drdpr(0, 0) = global_point(1) * sinp * cosr - global_point(2) * sinp * sinr;
      drdpr(1, 0) = -global_point(1) * sinr  - global_point(2) * cosr;
      drdpr(0, 1) = -global_point(0) * sinp + global_point(1) * cosp * sinr + global_point(2) * cosp * cosr;
      drdpr(1, 1) = 0.0;
      drdpr(0, 2) = -1;
      drdpr(1, 3) = -1;
      // std::cout << "jacobian2" << std::endl;
      // std::cout << drdpr << std::endl;
      drdlm = sqrt_info_ * drdpr;
    }
  }
  return true;
}


void LaserEdgeFactor::CheckJacobian(double const *const *parameters) {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Vector3d global_point = Qi * local_point_ + Pi;

  LineInfo line_info(Eigen::Vector4d(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]));
  auto normal = line_info.get_normal();
  auto center = line_info.get_center();
  Eigen::Matrix3d dmat = (Eigen::Matrix3d::Identity() - normal * normal.transpose());

  Eigen::Matrix3d Rzv = line_info.Rvz().transpose();
  Eigen::Vector3d error = line_info.distance(global_point);
  double sinr = std::sin(parameters[1][0]), cosr = std::cos(parameters[1][0]);
  double sinp = std::sin(parameters[1][1]), cosp = std::cos(parameters[1][1]);

  std::cout << "LaserEdgeFactor error norm: " << error.norm() << std::endl;
  std::cout << "LaserEdgeFactor residual norm: " << (Rzv.block<2, 3>(0, 0) * error).norm() << std::endl;

  Eigen::Vector2d residual = sqrt_info_ * Rzv.block<2, 3>(0, 0) * error;
  std::cout << "LaserEdgeFactor Residual: " << residual.norm() << std::endl;
  
  Eigen::Matrix<double, 2, 7, Eigen::RowMajor> drdT;
  drdT.setZero();
  Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dqdT;
  dqdT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  dqdT.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(local_point_);
  drdT.block<2, 6>(0, 0) = sqrt_info_ * Rzv.block<2, 3>(0, 0) * dmat * dqdT;

  Eigen::Matrix<double, 2, 4, Eigen::RowMajor> drdlm;
  drdlm.setZero();
  Eigen::Matrix<double, 3, 6> dXdnc;
  Eigen::Vector3d vec = global_point - center;
  dXdnc.leftCols<3>() = -(Eigen::Matrix3d::Identity() * (normal.transpose() * vec) + vec * normal.transpose());
  dXdnc.rightCols<3>() = -dmat;

  Eigen::Matrix<double, 2, 2> drdpr;
  drdpr(0, 0) = error(1) * sinp * cosr - error(2) * sinp * sinr;
  drdpr(1, 0) = -error(1) * sinr - error(2) * cosr;
  drdpr(0, 1) = -error(0) * sinp + error(1) * cosp * sinr + error(2) * cosp * cosr;
  drdpr(1, 1) = 0.0;

  drdlm = Rzv.block<2, 3>(0, 0) * dXdnc * line_info.jacobian();
  drdlm.block<2, 2>(0, 0) += drdpr;
  drdlm = sqrt_info_ * drdlm;


  // Eigen::Matrix<double, 2, 4, Eigen::RowMajor> drdlm;
  // drdlm.setZero();

  // Eigen::Matrix<double, 2, 2> drdpr;
  // drdpr(0, 0) = global_point(1) * sinp * cosr - global_point(2) * sinp * sinr;
  // drdpr(1, 0) = -global_point(1) * sinr - global_point(2) * cosr;
  // drdpr(0, 1) = -global_point(0) * sinp + global_point(1) * cosp * sinr + global_point(2) * cosp * cosr;
  // drdpr(1, 1) = 0.0;

  // drdlm.block<2, 2>(0, 0) = drdpr;
  // drdlm(0, 2) = -1;
  // drdlm(1, 3) = -1;
  // drdlm = sqrt_info_ * drdlm;

  // turb line
  std::cout << "[CheckJacobian] Turb Line" << std::endl;
  {
    Eigen::Matrix<double, 4, 1> delta = Eigen::Matrix<double, 4, 1>::Random() * 0.001;
    LineInfo line_info_turb(Eigen::Vector4d(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]) + delta);
    Eigen::Vector2d residual_turb = sqrt_info_ * Rzv.block<2, 3>(0, 0) * line_info_turb.distance(global_point);
    std::cout << "residual         err: " << (residual_turb - residual).transpose() << std::endl;
    std::cout << "jacobian * delta err: " << (drdlm * delta).transpose() << std::endl;
    double err_ratio = std::abs((residual_turb - residual).norm() - (drdlm * delta).norm()) * 100 / (residual_turb - residual).norm();
    std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    assert(err_ratio < 4.0);
  }

  // turb pose
  std::cout << "[CheckJacobian] Turb Pose" << std::endl;
  {
    Eigen::Vector3d delta_p = Eigen::Vector3d::Random() * 0.1;
    Eigen::Vector3d delta_r = Eigen::Vector3d::Random() * 0.03;
    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = delta_p;
    delta.tail<3>() = delta_r;
    Eigen::Quaterniond Qi_turb = Qi * deltaQ(delta_r);
    Eigen::Vector3d Pi_turb = Pi + delta_p;
    Eigen::Vector3d global_point_turb = Qi_turb * local_point_ + Pi_turb;
    Eigen::Vector2d residual_turb = sqrt_info_ * Rzv.block<2, 3>(0, 0) * line_info.distance(global_point_turb);
    std::cout << "residual         err: " << (residual_turb - residual).transpose() << std::endl;
    std::cout << "jacobian * delta err: " << (drdT.leftCols<6>() * delta).transpose() << std::endl;
    double err_ratio = std::abs((residual_turb - residual).norm() - (drdT.leftCols<6>() * delta).norm()) * 100 / (residual_turb - residual).norm();
    std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    assert(err_ratio < 4.0);
  }
}



bool LaserEdge2PFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Vector3d gpa = Qi * pa_ + Pi;
  Eigen::Vector3d gpb = Qi * pb_ + Pi;

  LineInfo line_info(Eigen::Vector4d(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]));
  auto normal = line_info.get_normal();
  auto center = line_info.get_center();

  Eigen::Vector3d ea = line_info.distance(gpa);
  Eigen::Vector3d eb = line_info.distance(gpb);

  Eigen::Matrix<double, 2, 3> Rzv = line_info.Rvz().transpose().block<2, 3>(0, 0);
  double sinr = std::sin(parameters[1][0]), cosr = std::cos(parameters[1][0]);
  double sinp = std::sin(parameters[1][1]), cosp = std::cos(parameters[1][1]);

  Eigen::Map<Eigen::Matrix<double, 4, 1>> residual(residuals);
  residual.head<2>() = Rzv * ea;
  residual.tail<2>() = Rzv * eb;
  residual = sqrt_info_ * residual;

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dqadT;
      dqadT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      dqadT.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(pa_);

      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dqbdT;
      dqbdT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      dqbdT.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(pb_);

      drdT.block<2, 6>(0, 0) = Rzv * dqadT;
      drdT.block<2, 6>(2, 0) = Rzv * dqbdT;
      drdT = sqrt_info_ * drdT;
    }
    if(jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> drdlm(jacobians[1]);
      drdlm.setZero();

      Eigen::Matrix<double, 2, 4> dradpr;
      dradpr.setZero();
      dradpr(0, 0) = gpa(1) * sinp * cosr - gpa(2) * sinp * sinr;
      dradpr(1, 0) = -gpa(1) * sinr - gpa(2) * cosr;
      dradpr(0, 1) = -gpa(0) * sinp + gpa(1) * cosp * sinr + gpa(2) * cosp * cosr;
      dradpr(1, 1) = 0.0;
      dradpr(0, 2) = -1;
      dradpr(1, 3) = -1;

      Eigen::Matrix<double, 2, 4> drbdpr;
      drbdpr.setZero();
      drbdpr(0, 0) = gpb(1) * sinp * cosr - gpb(2) * sinp * sinr;
      drbdpr(1, 0) = -gpb(1) * sinr - gpb(2) * cosr;
      drbdpr(0, 1) = -gpb(0) * sinp + gpb(1) * cosp * sinr + gpb(2) * cosp * cosr;
      drbdpr(1, 1) = 0.0;
      drbdpr(0, 2) = -1;
      drbdpr(1, 3) = -1;

      drdlm.block<2, 4>(0, 0) = dradpr;
      drdlm.block<2, 4>(2, 0) = drbdpr;
      drdlm = sqrt_info_ * drdlm;
    }
  }
  return true;
}

void LaserEdge2PFactor::CheckJacobian(double const *const *parameters) {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Vector3d gpa = Qi * pa_ + Pi;
  Eigen::Vector3d gpb = Qi * pb_ + Pi;

  LineInfo line_info(Eigen::Vector4d(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]));
  auto normal = line_info.get_normal();
  auto center = line_info.get_center();

  Eigen::Vector3d ea = line_info.distance(gpa);
  Eigen::Vector3d eb = line_info.distance(gpb);

  Eigen::Matrix<double, 2, 3> Rzv = line_info.Rvz().transpose().block<2, 3>(0, 0);
  double sinr = std::sin(parameters[1][0]), cosr = std::cos(parameters[1][0]);
  double sinp = std::sin(parameters[1][1]), cosp = std::cos(parameters[1][1]);

  Eigen::Matrix<double, 4, 1> residual;
  residual.head<2>() = Rzv * ea;
  residual.tail<2>() = Rzv * eb;

  // Jacobian Pose
  Eigen::Matrix<double, 4, 7> drdT;
  drdT.setZero();
  
  Eigen::Matrix<double, 3, 6> dqadT;
  dqadT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  dqadT.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(pa_);

  Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dqbdT;
  dqbdT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  dqbdT.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(pb_);

  drdT.block<2, 6>(0, 0) = Rzv * dqadT;
  drdT.block<2, 6>(2, 0) = Rzv * dqbdT;

  // Jacobian Line
  Eigen::Matrix<double, 4, 4> drdlm;
  drdlm.setZero();

  Eigen::Matrix<double, 2, 4> dradpr;
  dradpr.setZero();
  dradpr(0, 0) = gpa(1) * sinp * cosr - gpa(2) * sinp * sinr;
  dradpr(1, 0) = -gpa(1) * sinr - gpa(2) * cosr;
  dradpr(0, 1) = -gpa(0) * sinp + gpa(1) * cosp * sinr + gpa(2) * cosp * cosr;
  dradpr(1, 1) = 0.0;
  dradpr(0, 2) = -1;
  dradpr(1, 3) = -1;

  Eigen::Matrix<double, 2, 4> drbdpr;
  drbdpr.setZero();
  drbdpr(0, 0) = gpb(1) * sinp * cosr - gpb(2) * sinp * sinr;
  drbdpr(1, 0) = -gpb(1) * sinr - gpb(2) * cosr;
  drbdpr(0, 1) = -gpb(0) * sinp + gpb(1) * cosp * sinr + gpb(2) * cosp * cosr;
  drbdpr(1, 1) = 0.0;
  drbdpr(0, 2) = -1;
  drbdpr(1, 3) = -1;

  drdlm.block<2, 4>(0, 0) = dradpr;
  drdlm.block<2, 4>(2, 0) = drbdpr;


  // turb pose
  std::cout << "[CheckJacobian] Turb Pose" << std::endl;
  {
    Eigen::Vector3d delta_p = Eigen::Vector3d::Random() * 0.1;
    Eigen::Vector3d delta_r = Eigen::Vector3d::Random() * 0.03;
    Eigen::Matrix<double, 6, 1> delta;
    delta.head<3>() = delta_p;
    delta.tail<3>() = delta_r;
    Eigen::Quaterniond Qi_turb = Qi * deltaQ(delta_r);
    Eigen::Vector3d Pi_turb = Pi + delta_p;

    Eigen::Vector3d gpa_turb = Qi_turb * pa_ + Pi_turb;
    Eigen::Vector3d gpb_turb = Qi_turb * pb_ + Pi_turb;

    Eigen::Vector3d ea_turb = line_info.distance(gpa_turb);
    Eigen::Vector3d eb_turb = line_info.distance(gpb_turb);

    Eigen::Matrix<double, 4, 1> residual_turb;
    residual_turb.head<2>() = Rzv * ea_turb;
    residual_turb.tail<2>() = Rzv * eb_turb;

    std::cout << "residual         err: " << (residual_turb - residual).transpose() << std::endl;
    std::cout << "jacobian * delta err: " << (drdT.leftCols<6>() * delta).transpose() << std::endl;
    double err_ratio = std::abs((residual_turb - residual).norm() - (drdT.leftCols<6>() * delta).norm()) * 100 / (residual_turb - residual).norm();
    std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    assert(err_ratio < 4.0);
  }


  // turb line
  std::cout << "[CheckJacobian] Turb Line" << std::endl;
  {
    Eigen::Matrix<double, 4, 1> delta = Eigen::Matrix<double, 4, 1>::Random() * 0.02;
    LineInfo line_info_turb(Eigen::Vector4d(parameters[1][0], parameters[1][1], parameters[1][2], parameters[1][3]) + delta);

    Eigen::Matrix<double, 2, 3> Rzv_turb = line_info.Rvz().transpose().block<2, 3>(0, 0);
    double sinr_turb = std::sin(parameters[1][0] + delta(0)), cosr_turb = std::cos(parameters[1][0] + delta(0));
    double sinp_turb = std::sin(parameters[1][1] + delta(1)), cosp_turb = std::cos(parameters[1][1] + delta(1));

    Eigen::Vector3d ea_turb = line_info_turb.distance(gpa);
    Eigen::Vector3d eb_turb = line_info_turb.distance(gpb);

    Eigen::Matrix<double, 4, 1> residual_turb;
    residual_turb.head<2>() = Rzv_turb * ea_turb;
    residual_turb.tail<2>() = Rzv_turb * eb_turb;

    // Eigen::Vector2d residual_turb = sqrt_info_ * Rzv.block<2, 3>(0, 0) * line_info_turb.distance(global_point);
    std::cout << "residual         err: " << (residual_turb - residual).transpose() << std::endl;
    std::cout << "jacobian * delta err: " << (drdlm * delta).transpose() << std::endl;
    double err_ratio = std::abs((residual_turb - residual).norm() - (drdlm * delta).norm()) * 100 / (residual_turb - residual).norm();
    std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    assert(err_ratio < 4.0);
  }

}



bool LaserEdgeOnly2PFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

  Eigen::Vector3d gpa = Twb_ * pa_;
  Eigen::Vector3d gpb = Twb_ * pb_;

  LineInfo line_info(Eigen::Vector4d(parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3]));
  auto normal = line_info.get_normal();
  auto center = line_info.get_center();

  Eigen::Vector3d ea = line_info.distance(gpa);
  Eigen::Vector3d eb = line_info.distance(gpb);

  Eigen::Matrix<double, 2, 3> Rzv = line_info.Rvz().transpose().block<2, 3>(0, 0);
  double sinr = std::sin(parameters[0][0]), cosr = std::cos(parameters[0][0]);
  double sinp = std::sin(parameters[0][1]), cosp = std::cos(parameters[0][1]);

  Eigen::Map<Eigen::Matrix<double, 4, 1>> residual(residuals);
  residual.head<2>() = Rzv * ea;
  residual.tail<2>() = Rzv * eb;
  residual = sqrt_info_ * residual;

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>> drdlm(jacobians[0]);
      drdlm.setZero();

      Eigen::Matrix<double, 2, 4> dradpr;
      dradpr.setZero();
      dradpr(0, 0) = gpa(1) * sinp * cosr - gpa(2) * sinp * sinr;
      dradpr(1, 0) = -gpa(1) * sinr - gpa(2) * cosr;
      dradpr(0, 1) = -gpa(0) * sinp + gpa(1) * cosp * sinr + gpa(2) * cosp * cosr;
      dradpr(1, 1) = 0.0;
      dradpr(0, 2) = -1;
      dradpr(1, 3) = -1;

      Eigen::Matrix<double, 2, 4> drbdpr;
      drbdpr.setZero();
      drbdpr(0, 0) = gpb(1) * sinp * cosr - gpb(2) * sinp * sinr;
      drbdpr(1, 0) = -gpb(1) * sinr - gpb(2) * cosr;
      drbdpr(0, 1) = -gpb(0) * sinp + gpb(1) * cosp * sinr + gpb(2) * cosp * cosr;
      drbdpr(1, 1) = 0.0;
      drbdpr(0, 2) = -1;
      drbdpr(1, 3) = -1;

      drdlm.block<2, 4>(0, 0) = dradpr;
      drdlm.block<2, 4>(2, 0) = drbdpr;
      drdlm = sqrt_info_ * drdlm;
    }
  }
  return true;
}

}