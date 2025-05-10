#include "factor/laser_surf_factor.h"

namespace SLIM {
bool LaserPointToPointFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Map<Eigen::Vector3d> residual(residuals);
  residual = sqrt_info_ * (Qi * p_ + Pi - q_);

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      drdT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      drdT.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(p_);
      drdT = sqrt_info_ * drdT;
    }
  }
  return true;
}

bool LaserPointToSurfaceFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  residuals[0] = sqrt_info_ * n_.transpose() * (Qi * p_ + Pi - q_);

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      Eigen::Matrix<double, 3, 6, Eigen::RowMajor> dgp_dTi;
      dgp_dTi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
      dgp_dTi.block<3, 3>(0, 3) = -Qi.toRotationMatrix() * skewSymmetric(p_);
      drdT.block<1, 6>(0, 0) = sqrt_info_ * (n_.transpose() * dgp_dTi);
    }
  }
  return true;
}

bool LaserSurfPriorFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  SurfaceInfo surface(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]));
  // std::cout << "surface: " << surface.parameters().transpose() << std::endl;
  const auto normal = surface.get_normal();

  Eigen::Map<Eigen::Matrix<double, 4, 1>> residual(residuals);
  Eigen::Matrix<double, 4, 4> sqrt_info_mat = Eigen::Matrix<double, 4, 4>::Identity() * sqrt_info_;  

  residuals[0] = sqrt_info_ * surface.distance(pa_);
  residuals[1] = sqrt_info_ * surface.distance(pb_);
  residuals[2] = sqrt_info_ * surface.distance(pc_);
  residuals[3] = sqrt_info_ * surface.distance(pd_);
  // std::cout << "residual: " << residual.transpose() << std::endl;

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> drdlm(jacobians[0]);
      drdlm.setZero();
      Eigen::Matrix<double, 4, 4> dr_dnd;
      dr_dnd.block<1, 3>(0, 0) = pa_.transpose();
      dr_dnd(0, 3) = 1.0;
      dr_dnd.block<1, 3>(1, 0) = pb_.transpose();
      dr_dnd(1, 3) = 1.0;
      dr_dnd.block<1, 3>(2, 0) = pc_.transpose();
      dr_dnd(2, 3) = 1.0;
      dr_dnd.block<1, 3>(3, 0) = pd_.transpose();
      dr_dnd(3, 3) = 1.0;
      drdlm = sqrt_info_mat * dr_dnd * surface.jacobian();
    }
  }
  // printf("LaserSurfPriorFactor Evaluate Success!\n");
  return true;

}


void LaserSurfPriorFactor::CheckJacobian(double const *const *parameters) {
  SurfaceInfo surface(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]));
  auto normal = surface.get_normal();

  Eigen::Matrix<double, 4, 1> residual;
  Eigen::Matrix<double, 4, 4> sqrt_info_mat = Eigen::Matrix<double, 4, 4>::Identity() * sqrt_info_;  
  
  residual(0) = sqrt_info_ * surface.distance(pa_);
  residual(1) = sqrt_info_ * surface.distance(pb_);
  residual(2) = sqrt_info_ * surface.distance(pc_);
  residual(3) = sqrt_info_ * surface.distance(pd_);

  Eigen::Matrix<double, 4, 3, Eigen::RowMajor> drdlm;
  drdlm.setZero();
  Eigen::Matrix<double, 4, 4> dr_dnd;
  dr_dnd.block<1, 3>(0, 0) = pa_.transpose();
  dr_dnd(0, 3) = 1.0;
  dr_dnd.block<1, 3>(1, 0) = pb_.transpose();
  dr_dnd(1, 3) = 1.0;
  dr_dnd.block<1, 3>(2, 0) = pc_.transpose();
  dr_dnd(2, 3) = 1.0;
  dr_dnd.block<1, 3>(3, 0) = pd_.transpose();
  dr_dnd(3, 3) = 1.0;
  drdlm = sqrt_info_mat * dr_dnd * surface.jacobian();

  std::cout << "LaserSurfFactor Residual: " << residual.transpose() << std::endl;

  // turb surface
  std::cout << "[CheckJacobian] Turb Surface" << std::endl;
  {
    Eigen::Vector3d delta = Eigen::Vector3d::Random() * 0.02;
    Eigen::Vector3d vec_turb = Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]) + delta;
    SurfaceInfo surface_turb(vec_turb);
    Eigen::Vector4d residual_turb;
    residual_turb(0) = sqrt_info_ * surface_turb.distance(pa_);
    residual_turb(1) = sqrt_info_ * surface_turb.distance(pb_);
    residual_turb(2) = sqrt_info_ * surface_turb.distance(pc_);
    residual_turb(3) = sqrt_info_ * surface_turb.distance(pd_);

    // double residual_turb = sqrt_info_ * surface_turb.distance(global_point);
    std::cout << "residual err:         " << (residual_turb - residual).transpose() << std::endl;
    std::cout << "jacobian * delta err: " << (drdlm * delta).transpose() << std::endl;
    // double err_ratio = std::abs(std::abs(residual_turb - residual) - std::abs(drdlm * delta)) * 100 / std::abs(residual_turb - residual);
    // std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    // assert(err_ratio < 4.0);
  }
}

bool LaserSurfFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Vector3d global_point = Qi * local_point_ + Pi;

  SurfaceInfo surface(Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]));
  const auto normal = surface.get_normal();
  residuals[0] = sqrt_info_ * surface.distance(global_point);

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      drdT.block<1, 3>(0, 0) = normal.transpose();
      drdT.block<1, 3>(0, 3) = -normal.transpose() * Qi.toRotationMatrix() * skewSymmetric(local_point_);
      drdT = sqrt_info_ * drdT;
    }
    if(jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 1, 3, Eigen::RowMajor>> drdlm(jacobians[1]);
      drdlm.setZero();
      Eigen::Matrix<double, 1, 4> dr_dnd;
      dr_dnd.block<1, 3>(0, 0) = global_point.transpose();
      dr_dnd(0, 3) = 1.0;
      drdlm = sqrt_info_ * dr_dnd * surface.jacobian();
    }
  }
  return true;
}

void LaserSurfFactor::CheckResidual(double const *const *parameters, double *residuals) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Vector3d global_point = Qi * local_point_ + Pi;

  SurfaceInfo surface(Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]));
  const auto normal = surface.get_normal();
  residuals[0] = surface.distance(global_point);
  std::cout << "inside: Qwb: " << Qi.coeffs().transpose() << " global ob: " << global_point.transpose() << std::endl;
  std::cout << "inside: Twb: " << std::endl << Transform(Pi, Qi).matrix() << std::endl << " parameters: " << surface.parameters().transpose() << std::endl;  
}

void LaserSurfFactor::CheckJacobian(double const *const *parameters) {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Vector3d global_point = Qi * local_point_ + Pi;

  SurfaceInfo surface(Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]));
  auto normal = surface.get_normal();
  double residual = sqrt_info_ * surface.distance(global_point);

  Eigen::Matrix<double, 1, 7, Eigen::RowMajor> drdT;
  drdT.setZero();
  drdT.block<1, 3>(0, 0) = normal.transpose();
  drdT.block<1, 3>(0, 3) = -normal.transpose() * Qi.toRotationMatrix() * skewSymmetric(local_point_);
  drdT = sqrt_info_ * drdT;

  Eigen::Matrix<double, 1, 3, Eigen::RowMajor> drdlm;
  drdlm.setZero();
  Eigen::Matrix<double, 1, 4> dr_dnd;
  dr_dnd.block<1, 3>(0, 0) = global_point.transpose();
  dr_dnd(0, 3) = 1.0;
  drdlm = sqrt_info_ * dr_dnd * surface.jacobian();

  std::cout << "LaserSurfFactor Residual: " << residual << std::endl;

  // turb surface
  std::cout << "[CheckJacobian] Turb Surface" << std::endl;
  {
    Eigen::Vector3d delta = Eigen::Vector3d::Random() * 0.02;
    Eigen::Vector3d vec_turb = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]) + delta;
    SurfaceInfo surface_turb(vec_turb);
    double residual_turb = sqrt_info_ * surface_turb.distance(global_point);
    std::cout << "residual err:         " << residual_turb - residual << std::endl;
    std::cout << "jacobian * delta err: " << drdlm * delta << std::endl;
    double err_ratio = std::abs(std::abs(residual_turb - residual) - std::abs(drdlm * delta)) * 100 / std::abs(residual_turb - residual);
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
    double residual_turb = sqrt_info_ * surface.distance(global_point_turb);
    std::cout << "residual         err: " << residual_turb - residual << std::endl;
    std::cout << "jacobian * delta err: " << drdT.block<1, 6>(0, 0) * delta << std::endl;
    double err_ratio = std::abs(std::abs(residual_turb - residual) - std::abs(drdT.block<1, 6>(0, 0) * delta)) * 100 / std::abs(residual_turb - residual);
    std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    assert(err_ratio < 4.0);
  }
}

bool LaserSurf4PFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  
  Eigen::Vector3d gpa = Qi * vertices_[0] + Pi;
  Eigen::Vector3d gpb = Qi * vertices_[1] + Pi;
  Eigen::Vector3d gpc = Qi * vertices_[2] + Pi;
  Eigen::Vector3d gpd = Qi * vertices_[3] + Pi;

  SurfaceInfo surface(Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]));
  const auto normal = surface.get_normal().transpose();

  Eigen::Map<Eigen::Matrix<double, 4, 1>> residual(residuals);
  residual = sqrt_info_ * Eigen::Vector4d(surface.distance(gpa), surface.distance(gpb), surface.distance(gpc), surface.distance(gpd));

  Eigen::Matrix3d Rwb = Qi.toRotationMatrix();
  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 4, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      // drdT.block<1, 3>(0, 0) = sqrt_info_ * normal;
      // drdT.block<1, 3>(0, 3) = -sqrt_info_ * normal * Rwb * skewSymmetric(vertices_[0]);
      // drdT.block<1, 3>(1, 0) = sqrt_info_ * normal;
      // drdT.block<1, 3>(1, 3) = -sqrt_info_ * normal * Rwb * skewSymmetric(vertices_[1]);
      // drdT.block<1, 3>(2, 0) = sqrt_info_ * normal;
      // drdT.block<1, 3>(2, 3) = -sqrt_info_ * normal * Rwb * skewSymmetric(vertices_[2]);
      // drdT.block<1, 3>(3, 0) = sqrt_info_ * normal;
      // drdT.block<1, 3>(3, 3) = -sqrt_info_ * normal * Rwb * skewSymmetric(vertices_[3]);
      drdT.block<1, 3>(0, 0) = normal;
      drdT.block<1, 3>(0, 3) = -normal * Rwb * skewSymmetric(vertices_[0]);
      drdT.block<1, 3>(1, 0) = normal;
      drdT.block<1, 3>(1, 3) = -normal * Rwb * skewSymmetric(vertices_[1]);
      drdT.block<1, 3>(2, 0) = normal;
      drdT.block<1, 3>(2, 3) = -normal * Rwb * skewSymmetric(vertices_[2]);
      drdT.block<1, 3>(3, 0) = normal;
      drdT.block<1, 3>(3, 3) = -normal * Rwb * skewSymmetric(vertices_[3]);
      drdT = sqrt_info_ * drdT;
    }
    if(jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> drdlm(jacobians[1]);
      drdlm.setZero();
      Eigen::Matrix<double, 4, 4> drdnd;
      drdnd.block<1, 3>(0, 0) = gpa.transpose();
      drdnd(0, 3) = 1.0;
      drdnd.block<1, 3>(1, 0) = gpb.transpose();
      drdnd(1, 3) = 1.0;
      drdnd.block<1, 3>(2, 0) = gpc.transpose();
      drdnd(2, 3) = 1.0;
      drdnd.block<1, 3>(3, 0) = gpd.transpose();
      drdnd(3, 3) = 1.0;
      drdlm = sqrt_info_ * drdnd * surface.jacobian();
    }
  }
  return true;
}




bool LaserSurf3PFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  
  Eigen::Vector3d gpa = Qi * vertices_[0] + Pi;
  Eigen::Vector3d gpb = Qi * vertices_[1] + Pi;
  Eigen::Vector3d gpc = Qi * vertices_[2] + Pi;

  SurfaceInfo surface(Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]));
  const auto normal = surface.get_normal().transpose();

  Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
  residual = sqrt_info_ * Eigen::Vector3d(surface.distance(gpa), surface.distance(gpb), surface.distance(gpc));

  Eigen::Matrix3d Rwb = Qi.toRotationMatrix();
  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> drdT(jacobians[0]);
      drdT.setZero();
      drdT.block<1, 3>(0, 0) = normal;
      drdT.block<1, 3>(0, 3) = -normal * Rwb * skewSymmetric(vertices_[0]);
      drdT.block<1, 3>(1, 0) = normal;
      drdT.block<1, 3>(1, 3) = -normal * Rwb * skewSymmetric(vertices_[1]);
      drdT.block<1, 3>(2, 0) = normal;
      drdT.block<1, 3>(2, 3) = -normal * Rwb * skewSymmetric(vertices_[2]);
      drdT = sqrt_info_ * drdT;
    }
    if(jacobians[1] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> drdlm(jacobians[1]);
      drdlm.setZero();
      Eigen::Matrix<double, 3, 4> drdnd;
      drdnd.block<1, 3>(0, 0) = gpa.transpose();
      drdnd(0, 3) = 1.0;
      drdnd.block<1, 3>(1, 0) = gpb.transpose();
      drdnd(1, 3) = 1.0;
      drdnd.block<1, 3>(2, 0) = gpc.transpose();
      drdnd(2, 3) = 1.0;
      drdlm = sqrt_info_ * drdnd * surface.jacobian();
    }
  }
  return true;
}


void LaserSurf3PFactor::CheckJacobian(double const *const *parameters) {
  Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
  Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);
  Eigen::Matrix3d Ri = Qi.toRotationMatrix();
  
  Eigen::Vector3d gpa = Qi * vertices_[0] + Pi;
  Eigen::Vector3d gpb = Qi * vertices_[1] + Pi;
  Eigen::Vector3d gpc = Qi * vertices_[2] + Pi;

  SurfaceInfo surface(Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]));
  const auto normal = surface.get_normal().transpose();

  Eigen::Vector3d residual;
  residual = Eigen::Vector3d(surface.distance(gpa), surface.distance(gpb), surface.distance(gpc));

  Eigen::Matrix<double, 3, 7> drdT;
  drdT.setZero();
  drdT.block<1, 3>(0, 0) = normal;
  drdT.block<1, 3>(0, 3) = -normal * Ri * skewSymmetric(vertices_[0]);
  drdT.block<1, 3>(1, 0) = normal;
  drdT.block<1, 3>(1, 3) = -normal * Ri * skewSymmetric(vertices_[1]);
  drdT.block<1, 3>(2, 0) = normal;
  drdT.block<1, 3>(2, 3) = -normal * Ri * skewSymmetric(vertices_[2]);

  Eigen::Matrix<double, 3, 3> drdlm;
  drdlm.setZero();
  Eigen::Matrix<double, 3, 4> drdnd;
  drdnd.block<1, 3>(0, 0) = gpa.transpose();
  drdnd(0, 3) = 1.0;
  drdnd.block<1, 3>(1, 0) = gpb.transpose();
  drdnd(1, 3) = 1.0;
  drdnd.block<1, 3>(2, 0) = gpc.transpose();
  drdnd(2, 3) = 1.0;
  drdlm = drdnd * surface.jacobian();

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

    Eigen::Vector3d gpa_turb = Qi_turb * vertices_[0] + Pi_turb;
    Eigen::Vector3d gpb_turb = Qi_turb * vertices_[1] + Pi_turb;
    Eigen::Vector3d gpc_turb = Qi_turb * vertices_[2] + Pi_turb;

    Eigen::Vector3d residual_turb;
    residual_turb = Eigen::Vector3d(surface.distance(gpa_turb), surface.distance(gpb_turb), surface.distance(gpb_turb));

    std::cout << "residual         err: " << (residual_turb - residual).transpose() << std::endl;
    std::cout << "jacobian * delta err: " << (drdT.leftCols<6>() * delta).transpose() << std::endl;
    double err_ratio = std::abs((residual_turb - residual).norm() - (drdT.leftCols<6>() * delta).norm()) * 100 / (residual_turb - residual).norm();
    std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    assert(err_ratio < 4.0);
  }


  // turb surface
  std::cout << "[CheckJacobian] Turb Surface" << std::endl;
  {
    Eigen::Vector3d delta = Eigen::Vector3d::Random() * 0.02;
    Eigen::Vector3d vec_turb = Eigen::Vector3d(parameters[1][0], parameters[1][1], parameters[1][2]) + delta;
    SurfaceInfo surface_turb(vec_turb);
    Eigen::Vector3d residual_turb;
    residual_turb(0) = surface_turb.distance(gpa);
    residual_turb(1) = surface_turb.distance(gpb);
    residual_turb(2) = surface_turb.distance(gpc);

    std::cout << "residual err:         " << (residual_turb - residual).transpose() << std::endl;
    std::cout << "jacobian * delta err: " << (drdlm * delta).transpose() << std::endl;
    double err_ratio = std::abs((residual_turb - residual).norm() - (drdlm * delta).norm()) * 100 / (residual_turb - residual).norm();
    std::cout << "err ratio: " << err_ratio << "%" << std::endl;
    assert(err_ratio < 4.0);
  }


}


bool LaserSurfOnly3PFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

  
  Eigen::Vector3d gpa = Twb_ * vertices_[0];
  Eigen::Vector3d gpb = Twb_ * vertices_[1];
  Eigen::Vector3d gpc = Twb_ * vertices_[2];

  SurfaceInfo surface(Eigen::Vector3d(parameters[0][0], parameters[0][1], parameters[0][2]));
  const auto normal = surface.get_normal().transpose();

  Eigen::Map<Eigen::Matrix<double, 3, 1>> residual(residuals);
  residual = sqrt_info_ * Eigen::Vector3d(surface.distance(gpa), surface.distance(gpb), surface.distance(gpc));

  if(jacobians != nullptr) {
    if(jacobians[0] != nullptr) {
      Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> drdlm(jacobians[0]);
      drdlm.setZero();
      Eigen::Matrix<double, 3, 4> drdnd;
      drdnd.block<1, 3>(0, 0) = gpa.transpose();
      drdnd(0, 3) = 1.0;
      drdnd.block<1, 3>(1, 0) = gpb.transpose();
      drdnd(1, 3) = 1.0;
      drdnd.block<1, 3>(2, 0) = gpc.transpose();
      drdnd(2, 3) = 1.0;
      drdlm = sqrt_info_ * drdnd * surface.jacobian();
    }
  }
  return true;
}

}