#include "nfr_solver.h"
#include <omp.h>

namespace SLIM {

inline void checkNaNInf(const Eigen::MatrixXd& mat, const std::string& token) {
  if(mat.array().isNaN().any() || mat.array().isInf().any()) {
    printf("Error NAN & INF Values in\n");
    std::cout << "ERROR: " << token << std::endl;
  }     
}

inline Eigen::VectorXd sym2vec(const Eigen::MatrixXd& mat) {
  assert(mat.rows() == mat.cols());
  int len = mat.cols() * (mat.cols() + 1) / 2;
  Eigen::VectorXd vec(len);
  int index = 0;
  for(int i = 0; i < mat.rows(); ++i) {
    for(int j = i; j < mat.cols(); ++j) {
      vec(index++) = mat(i, j);
    }
  }
  return vec;
}

inline Eigen::MatrixXd vec2sym(const Eigen::VectorXd& vec) {
  int len = vec.rows();
  int size = (std::sqrt(8 * len + 1) - 1) / 2;
  Eigen::MatrixXd mat(size, size);
  int index = 0;
  for(int i = 0; i < size; ++i) {
    for(int j = i; j < size; ++j) {
      mat(i, j) = vec(index);
      mat(j, i) = vec(index);
      index++;
    }
  }
  return mat;
}

void NFRSolver::calcBaseMat(Eigen::MatrixXd& Q, Eigen::MatrixXd& PQ, Eigen::MatrixXd& PQl, Eigen::MatrixXd& Wl, Eigen::MatrixXd& Wf) {

  printGpuMemInfo();
  CuSMat dsD(Ainv_), dsP, dsVl(Vl_), dsPtVl;
  CuDMat ddQ, ddPQVf;

  CuSMat dsPtVlmVf;
  Eigen::SparseMatrix<double> PtVl, DVl;
  CuDMat ddPQ;
  {
    CuSMat dsC(C_), dsU(U_), dsUt(U_.transpose());
    CuSMat dsVf(Vf_);
    CuDMat ddVf(Vf_.toDense());
  
    CuSMat dsQinv = dsC - (dsUt * dsD * dsU);
    ddQ = CuDMat(dsQinv.toEigenSparse().toDense()).inverse();
    Q = ddQ.toEigenDense();
    dsQinv.release();

    Eigen::MatrixXd Ql = solveCholeskyCUDA(Q);
    CuDMat ddQl(Ql);
    dsP = dsD * dsU;
    CuDMat ddPQl = dsP * ddQl;
    PQl = ddPQl.toEigenDense();
    ddPQl.release();
    ddQl.release();

    dsPtVl = dsUt * dsD * dsVl; // P^T * Vl          [fxm]
    PtVl = dsPtVl.toEigenSparse();

    dsPtVlmVf = dsPtVl - dsVf;
    dsPtVl.release();

    CuSMat dsDVl = dsD * dsVl;
    DVl = dsDVl.toEigenSparse();
    dsDVl.release();

    CuDMat ddQVf = ddQ * ddVf;
    ddPQVf = dsP * ddQVf;  
    ddQVf.release();  
  }

  // dsD.release();
  // dsP.release();
  // ddQ.release();
  // ddPQVf.release();

  std::cout << "end of step1" << std::endl;
  printGpuMemInfo();

  {
    CuDMat ddVlt = CuDMat(Vl_.toDense()).transpose();
    CuDMat ddVltPQVf = ddVlt * ddPQVf; // [mxl] x [lxm] = [mxm]
    ddPQVf.release();
    std::cout << "0" << std::endl;
    printGpuMemInfo();

    CuDMat ddHm_schur_cross = ddVltPQVf + ddVltPQVf.transpose();
    ddVltPQVf.release();
    std::cout << "1" << std::endl;
    printGpuMemInfo();
    
    CuDMat ddDVl(DVl.toDense());
    // CuDMat ddVltDVl = ddVlt * ddDVl;
    CuDMat ddSum = ddVlt * ddDVl;
    ddDVl.release();
    ddVlt.release();
    std::cout << "2" << std::endl;
    printGpuMemInfo();

    CuDMat ddPtVltQPtVl;
    CuDMat ddPtVl(PtVl.toDense());
    matmulAtXA(ddQ, ddPtVl, ddPtVltQPtVl);
    ddSum.addSelf(ddPtVltQPtVl);
    ddPtVltQPtVl.release();
    ddPtVl.release();
    std::cout << "3" << std::endl;
    printGpuMemInfo();

    CuDMat ddVftQVf;
    CuDMat ddVf(Vf_.toDense());
    matmulAtXA(ddQ, ddVf, ddVftQVf);
    ddSum.addSelf(ddVftQVf);
    ddVftQVf.release();
    ddVf.release();
    std::cout << "4" << std::endl;
    printGpuMemInfo();

    ddSum.minusSelf(ddHm_schur_cross);
    ddHm_schur_cross.release();
    CuDMat ddRinv = CuDMat(Vm_.toDense()) - ddSum;
    ddSum.release();
    // CuDMat ddRinv = CuDMat(Vm_.toDense()) - (ddVltDVl + ddPtVltQPtVl + ddVftQVf - ddHm_schur_cross); // [mxm]
    std::cout << "5" << std::endl;
    printGpuMemInfo();

    ddHm_schur_cross.release();
    // ddVltDVl.release();
    // ddPtVltQPtVl.release();
    // ddVftQVf.release();

    std::cout << "6" << std::endl;
    printGpuMemInfo();

    CuDMat ddR = ddRinv.inverse();
    Eigen::MatrixXd Rl = solveCholeskyCUDA(ddR.toEigenDense());
    CuDMat ddRl(Rl);
    ddRinv.release();
    ddR.release();
    std::cout << "7" << std::endl;
    printGpuMemInfo();

    CuDMat ddPtVlmVfRl = dsPtVlmVf * ddRl;
    std::cout << "8" << std::endl;
    printGpuMemInfo();


    ddPQ = dsP * ddQ;
    dsP.release();
    std::cout << "9" << std::endl;
    printGpuMemInfo();

    CuDMat ddWl = dsD * (dsVl * ddRl) + ddPQ * ddPtVlmVfRl; // [lxm]
    ddRl.release();
    dsD.release();
    dsVl.release();

    std::cout << "10" << std::endl;
    printGpuMemInfo();

    CuDMat ddWf = ddQ * ddPtVlmVfRl;        // [fxm]
    ddQ.release();
    ddPtVlmVfRl.release();
    std::cout << "11" << std::endl;
    printGpuMemInfo();

    PQ = ddPQ.toEigenDense();
    Wl = ddWl.toEigenDense();
    Wf = -ddWf.toEigenDense();
    ddPQ.release();
    ddWl.release();
    ddWf.release();
  }

  std::cout << "end of step2" << std::endl;
  printGpuMemInfo();
}

void NFRSolver::calcBaseMatV2(Eigen::MatrixXd& Q, Eigen::MatrixXd& PQ, Eigen::MatrixXd& PQl, Eigen::MatrixXd& Wl, Eigen::MatrixXd& Wf) {
  printGpuMemInfo();
  Eigen::SparseMatrix<double> Qinv = C_ - U_.transpose() * Ainv_ * U_;
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> Qqr(Qinv);
  Q = Qqr.solve(Eigen::MatrixXd::Identity(Qinv.rows(), Qinv.cols()));
  Eigen::SparseMatrix<double> P = Ainv_ * U_;
  Eigen::MatrixXd Ql = solveCholeskyCUDA(Q);
  checkNaNInf(Ql, "Ql");
  // Eigen::MatrixXd Ql = Eigen::LLT<Eigen::MatrixXd>(Q).matrixL();
  printGpuMemInfo();

  PQl = P * Ql;                           // P * Ql            [lxf]
  Eigen::SparseMatrix<double> PtVl = P.transpose() * Vl_; // P^T * Vl          [fxm]
  Eigen::SparseMatrix<double> DVl = Ainv_ * Vl_;          // A^{-1} * Vl       [lxm]
  Eigen::MatrixXd QVf = Q * Vf_;                          // Q * Vf            [fxm]
  Eigen::MatrixXd PQVf = P * QVf;                         // P * Q * Vf        [lxm]
  // Eigen::SparseMatrix<double> QPtVl = Q * PtVl;                       // Q * P^T * Vl      [fxm]
  // Eigen::SparseMatrix<double> PQPtVl = P * QPtVl;                     // P * Q * P^T * Vl  [lxm]

  Eigen::MatrixXd VltDVl = Vl_.transpose() * DVl;
  CuDMat ddSum = CuDMat(VltDVl);

  Eigen::MatrixXd PtVltQPtVl = PtVl.transpose() * Q * PtVl;
  CuDMat ddPtVltQPtVl(PtVltQPtVl);
  ddSum.addSelf(ddPtVltQPtVl);
  ddPtVltQPtVl.release();
  printGpuMemInfo();

  Eigen::MatrixXd VftQVf = Vf_.transpose() * QVf;
  CuDMat ddVftQVf(VftQVf);
  ddSum.addSelf(ddVftQVf);
  ddVftQVf.release();
  printGpuMemInfo();
  
  Eigen::MatrixXd VltPQVf = Vl_.transpose() * PQVf; // [mxl] x [lxm] = [mxm]
  CuDMat ddVltPQVf(VltPQVf);
  CuDMat ddHschur = ddVltPQVf + ddVltPQVf.transpose();
  ddVltPQVf.release();
  ddSum.minusSelf(ddHschur);
  ddHschur.release();
  printGpuMemInfo();
  
  // Eigen::MatrixXd Hm_schur_cross = VltPQVf + VltPQVf.transpose(); // [mxm]
  // Eigen::MatrixXd Rinv = Vm_ - (Vl_.transpose() * DVl + PtVl.transpose() * Q * PtVl + Vf_.transpose() * QVf - Hm_schur_cross); // [mxm]
  // Eigen::MatrixXd R = CuDMat(Rinv).inverse().toEigenDense();

  CuDMat ddRinv = CuDMat(Vm_.toDense()) - ddSum;

  Eigen::MatrixXd Rinv = ddRinv.toEigenDense();
  checkNaNInf(Rinv, "Rinv");

  ddSum.release();
  printGpuMemInfo();

  Eigen::MatrixXd R = ddRinv.inverse().toEigenDense();
  Eigen::MatrixXd Rl = solveCholeskyCUDA(R);
  checkNaNInf(Rl, "Rl");
  ddRinv.release();
  printGpuMemInfo();

  // Eigen::MatrixXd Rl = Eigen::LLT<Eigen::MatrixXd>(R).matrixL();  
  std::cout << "here" << std::endl;
  Eigen::MatrixXd PtVlmVfRl = (PtVl - Vf_) * Rl;
  PQ = P * Q;
  // Wl = Ainv_ * (Vl_ * Rl) + PQ * PtVlmVfRl; // [lxm]
  // Wf = -Q * PtVlmVfRl;        // [fxm]

  printGpuMemInfo();
  Eigen::MatrixXd DVlRl = Ainv_ * (Vl_ * Rl);
  Eigen::MatrixXd dAinv = Ainv_.toDense();
  Eigen::MatrixXd dVl = Vl_.toDense();
  checkNaNInf(dAinv, "dAinv");
  checkNaNInf(dVl, "dVl");
  checkNaNInf(DVlRl, "DVlRl");

  CuDMat ddPtVlmVfRl(PtVlmVfRl);
  CuDMat ddWl = CuDMat(DVlRl) + CuDMat(PQ) * ddPtVlmVfRl;
  CuDMat ddWf = CuDMat(Q) * ddPtVlmVfRl;
  ddPtVlmVfRl.release();

  checkNaNInf(PQ, "PQ");
  checkNaNInf(PQl, "PQl");

  Wl = ddWl.toEigenDense();
  Wf = -ddWf.toEigenDense();

  checkNaNInf(Wl, "Wl");
  checkNaNInf(Wf, "Wf");

  ddWl.release();
  ddWf.release();
  printGpuMemInfo();
}

void NFRSolver::init() {
  marg_block_size_ = map_marg_frame_.size();
  marg_parameter_size_ = marg_block_size_ * POSE_BLOCK_SIZE;

  retain_frame_block_size_ = map_retain_frame_.size();
  retain_line_block_size_ = map_retain_line_.size();
  retain_surf_block_size_ = map_retain_surf_.size();

  retain_block_size_ = retain_frame_block_size_ + retain_line_block_size_ + retain_surf_block_size_;
  retain_parameter_size_ = retain_frame_block_size_ * POSE_BLOCK_SIZE + retain_line_block_size_ * LINE_BLOCK_SIZE + retain_surf_block_size_ * SURF_BLOCK_SIZE;

  printf("[NFRSolver::init] marg_block_size_: %ld\n", marg_block_size_);
  printf("[NFRSolver::init] marg_parameter_size_: %ld\n", marg_parameter_size_);
  printf("[NFRSolver::init] retain_frame_block_size_: %ld\n", retain_frame_block_size_);
  printf("[NFRSolver::init] retain_line_block_size_: %ld\n", retain_line_block_size_);
  printf("[NFRSolver::init] retain_surf_block_size_: %ld\n", retain_surf_block_size_);
  printf("[NFRSolver::init] retain_block_size_: %ld\n", retain_block_size_);
  printf("[NFRSolver::init] retain_parameter_size_: %ld\n", retain_parameter_size_);
}

void NFRSolver::evalFramePrior(const Frame::Ptr& frame, Eigen::MatrixXd& Jf, Transform& virtual_ob) {
  virtual_ob = frame->Twb();
  Eigen::Matrix<double, 6, 6> drdTi;
  drdTi.setZero();
  drdTi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  drdTi.block<3, 3>(3, 3) = Qleft(virtual_ob.q().inverse() * frame->Twb().q()).bottomRightCorner<3, 3>();
  Jf = drdTi;
}

void NFRSolver::evalFrameToFrame(const Frame::Ptr& fi, const Frame::Ptr& fj, Eigen::MatrixXd& Ji, Eigen::MatrixXd& Jj, Transform& virtual_ob) {

  Transform rel_pose = fi->Twb().inverse() * fj->Twb();
  Eigen::Quaterniond Qi = fi->Twb().q(), Qj = fj->Twb().q();
  Eigen::Vector3d Pi = fi->Twb().p(), Pj = fj->Twb().p();
  Eigen::Matrix3d Ri = Qi.toRotationMatrix(), Rj = Qj.toRotationMatrix();  
  Eigen::Vector3d Pij = Ri.transpose() * (Pj - Pi);
  Eigen::Quaterniond Qij = Qi.inverse() * Qj;

  Eigen::Matrix<double, 6, 6> drdTi;
  drdTi.setZero();
  Eigen::Matrix3d dtdti = -Ri.transpose();
  Eigen::Matrix3d dtdqi = skewSymmetric(Pij);
  Eigen::Matrix3d dqdqi = -(Qright(Qij) * Qleft(rel_pose.q().inverse())).bottomRightCorner<3, 3>();
  drdTi.block<3, 3>(0, 0) = dtdti;
  drdTi.block<3, 3>(0, 3) = dtdqi;
  drdTi.block<3, 3>(3, 3) = dqdqi;

  Eigen::Matrix<double, 6, 6> drdTj;
  drdTj.setZero();
  Eigen::Matrix3d drdtj = Ri.transpose();
  Eigen::Matrix3d drdqj = Qleft(rel_pose.q().inverse() * Qij).bottomRightCorner<3, 3>();
  drdTj.block<3, 3>(0, 0) = drdtj;
  drdTj.block<3, 3>(3, 3) = drdqj;

  Ji = drdTi;
  Jj = drdTj;
  virtual_ob = rel_pose;
}

// generate a virtual measurement (point to line)
// output two virtual observation point, and jacobian J(4x6, 4x4)
void NFRSolver::evalFrameToLine(const Frame::Ptr& frame, const LineLM::Ptr& line,
                                Eigen::MatrixXd& Jf, Eigen::MatrixXd& Jl, std::vector<Eigen::Vector3d>& virtual_ob, int& point_num) {

  auto normal = line->normal();
  auto centroid = line->centroid();
  LineInfo line_info = line->parameters();
  auto Rzv = line_info.Rvz().transpose();
  double sinr = std::sin(line_info.parameters()(2)), cosr = std::cos(line_info.parameters()(2));
  double sinp = std::sin(line_info.parameters()(3)), cosp = std::cos(line_info.parameters()(3));

  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d squared_sum{Eigen::Matrix3d::Zero()};
  auto obs = line->getAllObs();
  point_num = 0;
  for(auto &ob_iter: obs) {
    Frame::Ptr const f = ob_iter.first;
    LineOB::Ptr const ob = ob_iter.second;
    Transform T = f->constTwb();
    auto gpa = T * ob->point_a(), gpb = T * ob->point_b();
    sum += gpa * ob->point_num_ * 0.5;
    sum += gpb * ob->point_num_ * 0.5;
    squared_sum += gpa * gpa.transpose() * ob->point_num_ * 0.5;
    squared_sum += gpb * gpb.transpose() * ob->point_num_ * 0.5;
    point_num += ob->point_num_;
  }

  sum /= point_num;
  squared_sum.noalias() = squared_sum / point_num - sum * sum.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(squared_sum);
  auto lambda = saes.eigenvalues();

  auto gpa = centroid + normal * std::sqrt(lambda(2));
  auto gpb = centroid - normal * std::sqrt(lambda(2));

  const Transform Twbr = frame->Twb();

  auto lpa = Twbr.inverse() * gpa;
  auto lpb = Twbr.inverse() * gpb;

  const Eigen::Matrix3d Rwbr = Twbr.dcm();
  Eigen::Matrix3d dmat = (Eigen::Matrix3d::Identity() - normal * normal.transpose());

  // pa to pose
  Eigen::Matrix<double, 2, 6> dradT;
  dradT.setZero();
  Eigen::Matrix<double, 3, 6> dqadT;
  dqadT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  dqadT.block<3, 3>(0, 3) = -Rwbr * skewSymmetric(lpa);
  dradT = Rzv.block<2, 3>(0, 0) * dmat * dqadT;

  // pb to pose
  Eigen::Matrix<double, 2, 6> drbdT;
  drbdT.setZero();
  Eigen::Matrix<double, 3, 6> dqbdT;
  dqbdT.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
  dqbdT.block<3, 3>(0, 3) = -Rwbr * skewSymmetric(lpb);
  drbdT = Rzv.block<2, 3>(0, 0) * dmat * dqbdT;

  // pa to line
  Eigen::Matrix<double, 2, 4> dradlm;
  dradlm.setZero();
  dradlm(0, 0) = gpa(1) * sinp * cosr - gpa(2) * sinp * sinr;
  dradlm(1, 0) = -gpa(1) * sinr - gpa(2) * cosr;
  dradlm(0, 1) = -gpa(0) * sinp + gpa(1) * cosp * sinr + gpa(2) * cosp * cosr;
  dradlm(1, 1) = 0.0;
  dradlm(0, 2) = -1;
  dradlm(1, 3) = -1;

  // pb to line
  Eigen::Matrix<double, 2, 4> drbdlm;
  drbdlm.setZero();
  drbdlm(0, 0) = gpb(1) * sinp * cosr - gpb(2) * sinp * sinr;
  drbdlm(1, 0) = -gpb(1) * sinr - gpb(2) * cosr;
  drbdlm(0, 1) = -gpb(0) * sinp + gpb(1) * cosp * sinr + gpb(2) * cosp * cosr;
  drbdlm(1, 1) = 0.0;
  drbdlm(0, 2) = -1;
  drbdlm(1, 3) = -1;

  Eigen::Vector4d residual;
  residual.head<2>() = Rzv.block<2, 3>(0, 0) * line_info.distance(gpa);
  residual.tail<2>() = Rzv.block<2, 3>(0, 0) * line_info.distance(gpb);

  Jf.resize(4, 6);
  Jl.resize(4, 4);
  Jf.setZero();
  Jl.setZero();
  Jf.block<2, 6>(0, 0) = dradT;
  Jf.block<2, 6>(2, 0) = drbdT;
  Jl.block<2, 4>(0, 0) = dradlm;
  Jl.block<2, 4>(2, 0) = drbdlm;
  
  virtual_ob.push_back(lpa);
  virtual_ob.push_back(lpb);
}

void NFRSolver::evalFrameToSurf(const Frame::Ptr& frame, const SurfaceLM::Ptr& surface,
                                Eigen::MatrixXd& Jf, Eigen::MatrixXd& Jl, std::vector<Eigen::Vector3d>& virtual_ob, int& point_num) {

  const Transform Trm = frame->Twb().inverse() * frame->Twb();

  Eigen::Vector3d sum{Eigen::Vector3d::Zero()};
  Eigen::Matrix3d squared_sum{Eigen::Matrix3d::Zero()};
  auto obs = surface->getAllObs();
  point_num = 0;
  for(auto &ob_iter: obs) {
    Frame::Ptr const f = ob_iter.first;
    SurfaceOB::Ptr const ob = ob_iter.second;
    Transform T = f->constTwb();
    auto gpa = T * ob->vertices()[0], gpb = T * ob->vertices()[1], gpc = T * ob->vertices()[2];
    sum += gpa * ob->point_num_ / 3.0;
    sum += gpb * ob->point_num_ / 3.0;
    sum += gpc * ob->point_num_ / 3.0;
    squared_sum += gpa * gpa.transpose() * ob->point_num_ / 3.0;
    squared_sum += gpb * gpb.transpose() * ob->point_num_ / 3.0;
    squared_sum += gpc * gpc.transpose() * ob->point_num_ / 3.0;
    point_num += ob->point_num_;
  }

  sum /= point_num;
  squared_sum.noalias() = squared_sum / point_num - sum * sum.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(squared_sum);
  Eigen::Vector3d const& lambda = saes.eigenvalues();
  Eigen::Matrix3d const& umat = saes.eigenvectors();
  Eigen::Vector3d const& centroid = surface->centroid();

  Eigen::Quaterniond q;
  q.setFromTwoVectors(umat.col(0), surface->normal());
  Eigen::Vector3d d0 = q * umat.col(2), d1 = q * umat.col(1);

  Eigen::Vector3d const gpa = centroid + d0 * std::sqrt(lambda(2) * 1.7);
  Eigen::Vector3d const gpb = centroid - 0.5 * d0 * std::sqrt(lambda(2) * 1.7) + d1 * std::sqrt(lambda(1) * 1.7);
  Eigen::Vector3d const gpc = centroid - 0.5 * d0 * std::sqrt(lambda(2) * 1.7) - d1 * std::sqrt(lambda(1) * 1.7);

  const Transform Twbr = frame->Twb();
  const Eigen::Matrix3d Rwbr = Twbr.dcm();
  const SurfaceInfo surface_info = surface->parameters();
  Eigen::Vector3d normal = surface_info.get_normal();

  Eigen::Vector3d lpa = Twbr.inverse() * gpa;
  Eigen::Vector3d lpb = Twbr.inverse() * gpb;
  Eigen::Vector3d lpc = Twbr.inverse() * gpc;

  Eigen::Matrix<double, 3, 6> drdT;
  drdT.setZero();
  drdT.block<1, 3>(0, 0) = normal.transpose();
  drdT.block<1, 3>(0, 3) = -normal.transpose() * Rwbr * skewSymmetric(lpa);
  drdT.block<1, 3>(1, 0) = normal.transpose();
  drdT.block<1, 3>(1, 3) = -normal.transpose() * Rwbr * skewSymmetric(lpb);
  drdT.block<1, 3>(2, 0) = normal.transpose();
  drdT.block<1, 3>(2, 3) = -normal.transpose() * Rwbr * skewSymmetric(lpc);
  // drdT.block<1, 3>(3, 0) = normal.transpose();
  // drdT.block<1, 3>(3, 3) = -normal.transpose() * Rwbr * skewSymmetric(lpd);

  Eigen::Matrix<double, 3, 3> drdS;
  drdS.setZero();
  Eigen::Matrix<double, 3, 4> drdnd;
  drdnd.block<1, 3>(0, 0) = (Twbr * lpa).transpose();
  drdnd(0, 3) = 1.0;
  drdnd.block<1, 3>(1, 0) = (Twbr * lpb).transpose();
  drdnd(1, 3) = 1.0;
  drdnd.block<1, 3>(2, 0) = (Twbr * lpc).transpose();
  drdnd(2, 3) = 1.0;
  // drdnd.block<1, 3>(3, 0) = (Twbr * lpd).transpose();
  // drdnd(3, 3) = 1.0;
  drdS = drdnd * surface_info.jacobian();

  Eigen::Vector3d residual;
  residual(0) = surface_info.distance(gpa);
  residual(1) = surface_info.distance(gpb);
  residual(2) = surface_info.distance(gpc);

  Jf = drdT;
  Jl = drdS;

  virtual_ob.push_back(lpa);
  virtual_ob.push_back(lpb);
  virtual_ob.push_back(lpc);
}  

void NFRSolver::evalFactor(ceres::CostFunction* cost_func, ceres::LossFunction* loss_func, 
      const std::vector<double*>& parameter_blocks, Eigen::VectorXd& residuals, 
      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& jacobians) {
  residuals.resize(cost_func->num_residuals());
  std::vector<int> block_sizes = cost_func->parameter_block_sizes();
  double **raw_jacobians = new double *[block_sizes.size()];
  jacobians.resize(block_sizes.size());
  for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
    jacobians[i].resize(cost_func->num_residuals(), block_sizes[i]);
    raw_jacobians[i] = jacobians[i].data();
  }
  cost_func->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

  if (loss_func) {
    double residual_scaling_, alpha_sq_norm_;
    double sq_norm, rho[3];
    sq_norm = residuals.squaredNorm();
    loss_func->Evaluate(sq_norm, rho);

    double sqrt_rho1_ = sqrt(rho[1]);
    if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
      residual_scaling_ = sqrt_rho1_;
      alpha_sq_norm_ = 0.0;
    }
    else {
      const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
      const double alpha = 1.0 - sqrt(D);
      residual_scaling_ = sqrt_rho1_ / (1 - alpha);
      alpha_sq_norm_ = alpha / sq_norm;
    }
    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
      jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
    }
    residuals *= residual_scaling_;
  }
}

void NFRSolver::buildDenseStruct() {
  uint64_t total_size = marg_parameter_size_ + retain_parameter_size_;
  uint64_t total_block_size = retain_line_block_size_ + retain_surf_block_size_ + retain_frame_block_size_ + marg_block_size_;
  hessian_.resize(total_size, total_size);
  hessian_.setZero();

  TicToc timer;
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);

  for(auto fixed_frame: fixed_frames_) {
    auto fdim = getRetainFrameDim(fixed_frame, 0);
    hessian_.block<6, 6>(fdim, fdim).noalias() += Eigen::Matrix<double, 6, 6>::Identity() * 1e8;
  }

  // auto fdim = getRetainFrameDim(pivot_frame_, 0);
  // hessian_.block<6, 6>(fdim, fdim).noalias() += pivot_sqrt_info_;
  

  for(size_t i = 0; i < rel_pose_info_.size(); i++) {
    RelPoseInfo& info = rel_pose_info_[i];
    OrderingType fidim, fjdim;
    if(isMarginalized(info.fi)) {
      fidim = getMargFrameDim(info.fi);
    }
    else {
      fidim = getRetainFrameDim(info.fi);
    }
    if(isMarginalized(info.fj)) {
      fjdim = getMargFrameDim(info.fj);
    }
    else {
      fjdim = getRetainFrameDim(info.fj);
    }

    RelativePoseFactor *factor = new RelativePoseFactor(info.Tij, info.sqrt_info);
    std::vector<double*> parameter_blocks{info.fi->Twb().parameters().data(), info.fj->Twb().parameters().data()};
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
    evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

    Eigen::MatrixXd Hii, Hij, Hjj;
    Hii = (jacobians[0].transpose() * jacobians[0]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
    Hjj = (jacobians[1].transpose() * jacobians[1]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
    Hij = (jacobians[0].transpose() * jacobians[1]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);

    hessian_.block<6, 6>(fidim, fidim).noalias() += Hii;
    hessian_.block<6, 6>(fjdim, fjdim).noalias() += Hjj;
    hessian_.block<6, 6>(fidim, fjdim).noalias() += Hij;
    hessian_.block<6, 6>(fjdim, fidim).noalias() += Hij.transpose();
  }


  for(auto lm_iter: map_retain_line_) {
    const LineLM::Ptr lm = lm_iter.first;
    const OrderingType ldim = getLineDim(lm);
    auto obs = lm->getAllObs();
    for(auto &ob: obs) {
      Frame::Ptr frame = ob.first;
      OrderingType fdim, fid;
      if(isMarginalized(frame)) {
        fdim = getMargFrameDim(frame, 0);
      }
      else {
        fdim = getRetainFrameDim(frame, 0);
      }
      LaserEdge2PFactor *factor = new LaserEdge2PFactor(ob.second->point_a(), ob.second->point_b(), ob.second->sqrt_info());
      std::vector<double*> parameter_blocks{frame->Twb().parameters().data(), lm->parameters().data()};
      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
      Eigen::VectorXd residuals;
      evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

      Eigen::MatrixXd Hpp = (jacobians[0].transpose() * jacobians[0]);
      Eigen::MatrixXd Hll = (jacobians[1].transpose() * jacobians[1]);
      Eigen::MatrixXd Hlp = (jacobians[1].transpose() * jacobians[0]);

      hessian_.block<6, 6>(fdim, fdim).noalias() += Hpp.block<6, 6>(0, 0);
      hessian_.block<4, 4>(ldim, ldim).noalias() += Hll;
      hessian_.block<4, 6>(ldim, fdim).noalias() += Hlp.leftCols<6>();
      hessian_.block<6, 4>(fdim, ldim).noalias() += Hlp.leftCols<6>().transpose();

    }
  }
  for(auto lm_iter: map_retain_surf_) {
    const SurfaceLM::Ptr lm = lm_iter.first;
    const OrderingType sdim = getSurfDim(lm);
    auto obs = lm->getAllObs();
    for(auto &ob: obs) {
      Frame::Ptr frame = ob.first;
      OrderingType fdim, fid;
      if(isMarginalized(frame)) {
        fdim = getMargFrameDim(frame, 0);
      }
      else {
        fdim = getRetainFrameDim(frame, 0);
      }
 
      SurfaceOB::Ptr feature = ob.second;
      std::vector<Eigen::Vector3d> const vertices = feature->vertices();
      // LaserSurf4PFactor *factor = new LaserSurf4PFactor(vertices, feature->sqrt_info());
      LaserSurf3PFactor *factor = new LaserSurf3PFactor(vertices, feature->sqrt_info());

      std::vector<double*> parameter_blocks{frame->Twb().parameters().data(), lm->parameters().data()};
      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
      Eigen::VectorXd residuals;
      evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

      Eigen::MatrixXd Hpp = (jacobians[0].transpose() * jacobians[0]);
      Eigen::MatrixXd Hll = (jacobians[1].transpose() * jacobians[1]);
      Eigen::MatrixXd Hlp = (jacobians[1].transpose() * jacobians[0]);

      hessian_.block<6, 6>(fdim, fdim).noalias() += Hpp.block<6, 6>(0, 0);
      hessian_.block<3, 3>(sdim, sdim).noalias() += Hll;
      hessian_.block<3, 6>(sdim, fdim).noalias() += Hlp.leftCols<6>();
      hessian_.block<6, 3>(fdim, sdim).noalias() += Hlp.leftCols<6>().transpose();
    }
  }
  covariance_ = hessian_.inverse();

  uint64_t lm_dim = retain_line_block_size_ * LINE_BLOCK_SIZE + retain_surf_block_size_ * SURF_BLOCK_SIZE;
  uint64_t rf_dim = retain_frame_block_size_ * POSE_BLOCK_SIZE;
  uint64_t r_dim = lm_dim + rf_dim;
  uint64_t m_dim = marg_block_size_ * POSE_BLOCK_SIZE;

  Eigen::Ref<Eigen::MatrixXd> dense_Hrr = hessian_.block(0, 0, r_dim, r_dim);
  Eigen::Ref<Eigen::MatrixXd> dense_Hmm = hessian_.block(r_dim, r_dim, m_dim, m_dim);
  Eigen::Ref<Eigen::MatrixXd> dense_Hrm = hessian_.block(0, r_dim, r_dim, m_dim);
  marg_hessian_ = dense_Hrr - dense_Hrm * dense_Hmm.inverse() * dense_Hrm.transpose();
  marg_covariance_ = marg_hessian_.inverse();

  // std::cout << "Dense Hessian: " << std::endl << hessian_ << std::endl;

  printf("[NFRSolver] Building dense structure done! Time: %lf ms\n", timer.toc());
  MatrixVisualizer mvis;
  mvis.render(hessian_, "/home/summervibe/catkin_ws/src/uav_lsm/map_creator/img/DenseHessian.jpg");
}

void NFRSolver::buildSparseStruct() {
  printf("[NFRSolver] Start to build sparse structure!\n");
  uint64_t total_size = marg_parameter_size_ + retain_parameter_size_;
  std::map<uint64_t, std::map<uint64_t, std::vector<Eigen::MatrixXd>>> hessian_map; // row block id, col block id, Hkl
  uint64_t lm_block_size = retain_line_block_size_ + retain_surf_block_size_;
  uint64_t retain_block_size = retain_line_block_size_ + retain_surf_block_size_ + retain_frame_block_size_;
  uint64_t total_block_size = retain_line_block_size_ + retain_surf_block_size_ + retain_frame_block_size_ + marg_block_size_;

  uint64_t lm_dim = retain_line_block_size_ * LINE_BLOCK_SIZE + retain_surf_block_size_ * SURF_BLOCK_SIZE;
  uint64_t rf_dim = retain_frame_block_size_ * POSE_BLOCK_SIZE;
  uint64_t r_dim = lm_dim + rf_dim;
  uint64_t m_dim = marg_block_size_ * POSE_BLOCK_SIZE;

  A_.resize(lm_dim, lm_dim);
  U_.resize(lm_dim, rf_dim);
  C_.resize(rf_dim, rf_dim);
  Vl_.resize(lm_dim, m_dim);
  Vf_.resize(rf_dim, m_dim);
  Vm_.resize(m_dim, m_dim);

  printf("[NFRSolver] Evaluate full hessian ...\n");
  
  ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
  printf("[NFRSolver] Evaluate relative pose factors ...\n");

  omp_set_num_threads(8);
// #pragma omp parallel for shared(rel_pose_info_, hessian_map) 
  for(size_t i = 0; i < rel_pose_info_.size(); i++) {
    RelPoseInfo& info = rel_pose_info_[i];
    OrderingType fibdim, fjbdim;
    if(isMarginalized(info.fi)) {
      fibdim = getMargFrameBlockDim(info.fi);
    }
    else {
      fibdim = getRetainFrameBlockDim(info.fi);
    }
    if(isMarginalized(info.fj)) {
      fjbdim = getMargFrameBlockDim(info.fj);
    }
    else {
      fjbdim = getRetainFrameBlockDim(info.fj);
    }

    RelativePoseFactor *factor = new RelativePoseFactor(info.Tij, info.sqrt_info);
    std::vector<double*> parameter_blocks{info.fi->Twb().parameters().data(), info.fj->Twb().parameters().data()};
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
    evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

    Eigen::MatrixXd Hii, Hij, Hjj;
    Hii = (jacobians[0].transpose() * jacobians[0]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
    Hjj = (jacobians[1].transpose() * jacobians[1]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
    Hij = (jacobians[0].transpose() * jacobians[1]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);

    hessian_map[fibdim][fibdim].push_back(Hii);
    hessian_map[fjbdim][fjbdim].push_back(Hjj);
    if(fibdim < fjbdim) {
      hessian_map[fibdim][fjbdim].push_back(Hij);      
    }
    else if(fibdim > fjbdim) {
      hessian_map[fjbdim][fibdim].push_back(Hij.transpose());  
    }
    else {
      printf("[NFRSolver] Warning! There is an relative pose measurement between two same frames!\n");
    }

  }
  printf("[NFRSolver] Evaluate line factors ...\n");

  std::vector<LineLM::Ptr> retain_line_lms;
  for (auto const& element : map_retain_line_) {
    retain_line_lms.push_back(element.first);
  }
#pragma omp parallel for shared(retain_line_lms, hessian_map)   
  for(int i = 0; i < retain_line_lms.size(); ++i) {
    const LineLM::Ptr lm = retain_line_lms[i];
    const OrderingType ldim = getLineDim(lm);
    const OrderingType lbdim = getLineBlockDim(lm);

    omp_set_lock(&lock);
    auto obs = lm->getAllObs();
    for(auto &ob: obs) {
      Frame::Ptr frame = ob.first;
      OrderingType fdim, fbdim, fid;
      if(isMarginalized(frame)) {
        fdim = getMargFrameDim(frame);
        fbdim = getMargFrameBlockDim(frame);
      }
      else {
        fdim = getRetainFrameDim(frame);
        fbdim = getRetainFrameBlockDim(frame);
      }
      LaserEdge2PFactor *factor = new LaserEdge2PFactor(ob.second->point_a(), ob.second->point_b(), ob.second->sqrt_info());
      std::vector<double*> parameter_blocks{frame->Twb().parameters().data(), lm->parameters().data()};
      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
      Eigen::VectorXd residuals;
      evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

      Eigen::MatrixXd Hpp, Hll, Hlp;
      Hpp = (jacobians[0].transpose() * jacobians[0]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
      Hll = (jacobians[1].transpose() * jacobians[1]);
      Hlp = (jacobians[1].transpose() * jacobians[0]).leftCols<POSE_BLOCK_SIZE>();

      hessian_map[lbdim][lbdim].push_back(Hll);
      hessian_map[fbdim][fbdim].push_back(Hpp);
      hessian_map[lbdim][fbdim].push_back(Hlp);
    }
    omp_unset_lock(&lock);
  }

  printf("[NFRSolver] Evaluate surf factors ...\n");
  std::vector<SurfaceLM::Ptr> retain_surf_lms;
  for (auto const& element : map_retain_surf_) {
    retain_surf_lms.push_back(element.first);
  }

#pragma omp parallel for shared(retain_surf_lms, hessian_map) 
  for(int i = 0; i < retain_surf_lms.size(); ++i) {
    const SurfaceLM::Ptr lm = retain_surf_lms[i];
    const OrderingType sdim = getSurfDim(lm);
    const OrderingType sbdim = getSurfBlockDim(lm);
    auto obs = lm->getAllObs();
    omp_set_lock(&lock);
    for(auto &ob: obs) {
      Frame::Ptr frame = ob.first;
      OrderingType fdim, fbdim, fid;
      if(isMarginalized(frame)) {
        fdim = getMargFrameDim(frame);
        fbdim = getMargFrameBlockDim(frame);
      }
      else {
        fdim = getRetainFrameDim(frame);
        fbdim = getRetainFrameBlockDim(frame);
      }
 
      SurfaceOB::Ptr feature = ob.second;
      std::vector<Eigen::Vector3d> const vertices = feature->vertices();
      // LaserSurf4PFactor *factor = new LaserSurf4PFactor(vertices, feature->sqrt_info());
      LaserSurf3PFactor *factor = new LaserSurf3PFactor(vertices, feature->sqrt_info());
      std::vector<double*> parameter_blocks{frame->Twb().parameters().data(), lm->parameters().data()};
      std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
      Eigen::VectorXd residuals;
      evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

      Eigen::MatrixXd Hpp, Hll, Hlp;
      Hpp = (jacobians[0].transpose() * jacobians[0]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
      Hll = (jacobians[1].transpose() * jacobians[1]);
      Hlp = (jacobians[1].transpose() * jacobians[0]).leftCols<POSE_BLOCK_SIZE>();

      hessian_map[sbdim][sbdim].push_back(Hll);
      hessian_map[fbdim][fbdim].push_back(Hpp);
      hessian_map[sbdim][fbdim].push_back(Hlp);
    }
    omp_unset_lock(&lock);
  }
  printf("[NFRSolver] Evaluate prior factors ...\n");
  // for(auto fixed_frame: fixed_frames_) {
  //   auto fbdim = getRetainFrameBlockDim(fixed_frame);
  //   std::cout << "fbdim: " << fbdim << std::endl;
  //   hessian_map[fbdim][fbdim].push_back(Eigen::MatrixXd::Identity(POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) * 1e8);
  // }

  auto fbdim = getRetainFrameBlockDim(pivot_frame_);
  // hessian_map[fbdim][fbdim].push_back(pivot_sqrt_info_);
  hessian_map[fbdim][fbdim].push_back(Eigen::MatrixXd::Identity(POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) * 1e8);

  printf("[NFRSolver] Fill in the sparse blocks ...\n");
  A_blocks_.clear();
  std::vector<Eigen::Triplet<double>> U_trip, C_trip, Vl_trip, Vf_trip, Vm_trip;

  std::vector<Eigen::MatrixXd> A_line_blocks;
  std::vector<Eigen::MatrixXd> A_surf_blocks;
// #pragma omp parallel for shared(A_blocks_, C_, U_, Vl_, Vf_, Vm_, hessian_map) 
  for(auto row_iter = hessian_map.begin(); row_iter != hessian_map.end(); row_iter++) {
    int row_block_id = row_iter->first;
    for(auto col_iter = row_iter->second.begin(); col_iter != row_iter->second.end(); col_iter++) {
      int col_block_id = col_iter->first;
      if(row_block_id < lm_block_size && col_block_id < lm_block_size) { // A block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        A_blocks_.push_back(col_iter->second.front());

        if(row_block_id < retain_line_block_size_) {
          A_line_blocks.push_back(col_iter->second.front());
        }
        else {
          A_surf_blocks.push_back(col_iter->second.front());
        }
      }
      else if(row_block_id < lm_block_size && col_block_id >= lm_block_size && col_block_id < retain_block_size) { // U block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
        if(row_block_id < retain_line_block_size_) {
          int row_offset = row_block_id * LINE_BLOCK_SIZE;
          int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
          for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              U_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
            }
          }
        }
        else {
          int row_offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (row_block_id - retain_line_block_size_) * SURF_BLOCK_SIZE;
          int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
          for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              U_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
            }
          }
        }
      }
      else if(row_block_id >= lm_block_size && row_block_id < retain_block_size && col_block_id >= lm_block_size && col_block_id < retain_block_size) { // C block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
        int row_offset = (row_block_id - lm_block_size) * POSE_BLOCK_SIZE;
        int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
        for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
          for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
            C_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
          }
        }
        if(row_block_id != col_block_id) {
          for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              C_trip.push_back(Eigen::Triplet<double>(col_offset + i, row_offset + j, block(j, i)));
            }
          }          
        }
      }
      else if(row_block_id < lm_block_size && col_block_id >= retain_block_size && col_block_id < total_block_size) { // Vl block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
        if(row_block_id < retain_line_block_size_) {
          int row_offset = row_block_id * LINE_BLOCK_SIZE;
          int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
          for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              Vl_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
            }
          }
        }
        else {
          int row_offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (row_block_id - retain_line_block_size_) * SURF_BLOCK_SIZE;
          int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
          for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              Vl_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
            }
          }
        }
      }
      else if(row_block_id >= lm_block_size && row_block_id < retain_block_size && col_block_id >= retain_block_size && col_block_id < total_block_size) { // Vf block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
        int row_offset = (row_block_id - lm_block_size) * POSE_BLOCK_SIZE;
        int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
        for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
          for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
            Vf_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
          }
        }
      }
      else if(row_block_id >= retain_block_size && row_block_id < total_block_size && col_block_id >= retain_block_size && col_block_id < total_block_size) { // Hmm block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
        int row_offset = (row_block_id - retain_block_size) * POSE_BLOCK_SIZE;
        int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
        for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
          for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
            Vm_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
          }
        }
        if(row_block_id != col_block_id) {
          for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              Vm_trip.push_back(Eigen::Triplet<double>(col_offset + i, row_offset + j, block(j, i)));
            }
          }
        }
      }
      else {
        printf("[NFRSolver] Warning! There is an error index matrix in NFR! (row_block_id: %d, col_block_id: %d)\n", row_block_id, col_block_id);
      }
    }
  }

  U_.setFromTriplets(U_trip.begin(), U_trip.end());
  C_.setFromTriplets(C_trip.begin(), C_trip.end());
  Vl_.setFromTriplets(Vl_trip.begin(), Vl_trip.end());
  Vf_.setFromTriplets(Vf_trip.begin(), Vf_trip.end());
  Vm_.setFromTriplets(Vm_trip.begin(), Vm_trip.end());

  U_.makeCompressed();
  C_.makeCompressed();
  Vl_.makeCompressed();
  Vf_.makeCompressed();
  Vm_.makeCompressed();

  // printf("[NFRSolver] Write diagonal blocks ...\n");
  // for(int i = 0; i < retain_line_block_size_; i++) {
  //   int offset = i * LINE_BLOCK_SIZE;
  //   writeBlock(A_, A_blocks_[i], offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE);
  // }
  // for(int i = 0; i < retain_surf_block_size_; i++) {
  //   int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + i * SURF_BLOCK_SIZE;
  //   writeBlock(A_, A_blocks_[i + retain_line_block_size_], offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE);
  // }

  // A_.makeCompressed();

  // Eigen::MatrixXd error;
  // Eigen::MatrixXd dense_A = A_.toDense();
  // error = dense_A - hessian_.block(0, 0, lm_dim, lm_dim);
  // printf("error A: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());
  
  // Eigen::MatrixXd dense_C = C_.toDense();
  // error = dense_C - hessian_.block(lm_dim, lm_dim, rf_dim, rf_dim);
  // printf("error C: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

  // Eigen::MatrixXd dense_Vm = Vm_.toDense();
  // error = dense_Vm - hessian_.block(lm_dim + rf_dim, lm_dim + rf_dim, m_dim, m_dim);
  // printf("error Vm: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

  // Eigen::MatrixXd dense_U = U_.toDense();
  // error = dense_U - hessian_.block(0, lm_dim, lm_dim, rf_dim);
  // printf("error U: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

  // Eigen::MatrixXd dense_Vl = Vl_.toDense();
  // error = dense_Vl - hessian_.block(0, r_dim, lm_dim, m_dim);
  // printf("error Vl: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

  // Eigen::MatrixXd dense_Vf = Vf_.toDense();
  // error = dense_Vf - hessian_.block(lm_dim, r_dim, rf_dim, m_dim);
  // printf("error Vf: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

  
  TicToc timer;
  printf("[NFRSolver] Compute block-wise inverse ...\n");
  Ainv_.resize(lm_dim, lm_dim);

  // cpu parallel
//   Sll_blocks_.resize(A_blocks_.size());
// #pragma omp parallel for shared(A_blocks_, Sll_blocks_) 
//   for(int i = 0; i < A_blocks_.size(); i++) {
//     if(i < retain_line_block_size_) {
//       int offset = i * LINE_BLOCK_SIZE;
//       Eigen::MatrixXd inv_block = A_blocks_[i].inverse();
//       Sll_blocks_[i] = inv_block;
//       // writeBlock(Ainv_, inv_block, offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE);
//     }
//     else {
//       int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (i - retain_line_block_size_) * SURF_BLOCK_SIZE;
//       Eigen::MatrixXd inv_block = A_blocks_[i].inverse();
//       Sll_blocks_[i] = inv_block;
//       // writeBlock(Ainv_, inv_block, offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE);
//     }
//   }
//   for(int i = 0; i < A_blocks_.size(); i++) {
//     if(i < retain_line_block_size_) {
//       int offset = i * LINE_BLOCK_SIZE;
//       writeBlock(Ainv_, Sll_blocks_[i], offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE);
//     }
//     else {
//       int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (i - retain_line_block_size_) * SURF_BLOCK_SIZE;
//       writeBlock(Ainv_, Sll_blocks_[i], offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE);
//     }
//   }

  // gpu parallel
  std::vector<Eigen::MatrixXd> Sll_line_blocks, Sll_surf_blocks;
  if(A_line_blocks.size() > 0) {
    Sll_line_blocks = inverseParallelCUDA(A_line_blocks, 4);
  }
  if(A_surf_blocks.size() > 0) {
    Sll_surf_blocks = inverseParallelCUDA(A_surf_blocks, 3);
  }

  std::vector<Eigen::Triplet<double>> Ainv_trip;
  for(int index = 0; index < Sll_line_blocks.size(); index++) {
    int offset = index * LINE_BLOCK_SIZE;
    for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
      for(int j = 0; j < LINE_BLOCK_SIZE; j++) {
        Ainv_trip.push_back(Eigen::Triplet<double>(offset + i, offset + j, Sll_line_blocks[index](i, j)));
      }
    }
  }
  for(int index = 0; index < Sll_surf_blocks.size(); index++) {
    int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + index * SURF_BLOCK_SIZE;
    for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
      for(int j = 0; j < SURF_BLOCK_SIZE; j++) {
        Ainv_trip.push_back(Eigen::Triplet<double>(offset + i, offset + j, Sll_surf_blocks[index](i, j)));
      }
    }
  }
  Sll_blocks_.clear();
  Sll_blocks_.insert(Sll_blocks_.end(), Sll_line_blocks.begin(), Sll_line_blocks.end());
  Sll_blocks_.insert(Sll_blocks_.end(), Sll_surf_blocks.begin(), Sll_surf_blocks.end());

  Ainv_.setFromTriplets(Ainv_trip.begin(), Ainv_trip.end());
  Ainv_.makeCompressed();

  printf("[NFRSolver] Evaluate sparse covariance ...\n");
  timer.tic();

  // cpu version
  Eigen::SparseMatrix<double> Qinv = C_ - U_.transpose() * Ainv_ * U_;
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> Qqr(Qinv);
  Eigen::MatrixXd Q = Qqr.solve(Eigen::MatrixXd::Identity(Qinv.rows(), Qinv.cols()));
  Eigen::SparseMatrix<double> P = Ainv_ * U_;
  Eigen::MatrixXd Ql = Eigen::LLT<Eigen::MatrixXd>(Q).matrixL();

  Eigen::MatrixXd PQl = P * Ql;                           // P * Ql            [lxf]
  Eigen::SparseMatrix<double> PtVl = P.transpose() * Vl_; // P^T * Vl          [fxm]
  Eigen::SparseMatrix<double> DVl = Ainv_ * Vl_;          // A^{-1} * Vl       [lxm]
  Eigen::MatrixXd QVf = Q * Vf_;                          // Q * Vf            [fxm]
  Eigen::MatrixXd PQVf = P * QVf;                         // P * Q * Vf        [lxm]
  Eigen::MatrixXd QPtVl = Q * PtVl;                       // Q * P^T * Vl      [fxm]
  Eigen::MatrixXd PQPtVl = P * QPtVl;                     // P * Q * P^T * Vl  [lxm]

  Eigen::MatrixXd VltPQVf = Vl_.transpose() * PQVf; // [mxl] x [lxm] = [mxm]
  Eigen::MatrixXd Hm_schur_cross = VltPQVf + VltPQVf.transpose(); // [mxm]
  Eigen::MatrixXd Rinv = Vm_ - (Vl_.transpose() * DVl + PtVl.transpose() * Q * PtVl + Vf_.transpose() * QVf - Hm_schur_cross); // [mxm]
  checkNaNInf(Rinv, "Rinv");
  Eigen::MatrixXd R = Rinv.inverse();
  checkNaNInf(R, "R");
  Eigen::MatrixXd Rl = Eigen::LLT<Eigen::MatrixXd>(R).matrixL();  
  checkNaNInf(Rl, "Rl");
  Eigen::MatrixXd PtVlmVfRl = (PtVl - Vf_) * Rl;
  Eigen::MatrixXd PQ = P * Q;
  Eigen::MatrixXd Wl = Ainv_ * (Vl_ * Rl) + PQ * PtVlmVfRl; // [lxm]
  Eigen::MatrixXd Wf = -Q * PtVlmVfRl;        // [fxm]

  // Eigen::MatrixXd Q, PQ, PQl, Wl, Wf;
  // calcBaseMat(Q, PQ, PQl, Wl, Wf);
  // calcBaseMatV2(Q, PQ, PQl, Wl, Wf);
  
//   timer.tic();
#pragma omp parallel for shared(PQl, Wl, Sll_blocks_) 
  for(int i = 0; i < retain_line_block_size_; i++) {
    int offset = i * LINE_BLOCK_SIZE;
    Eigen::Ref<Eigen::MatrixXd> PQl_block = PQl.block(offset, 0, LINE_BLOCK_SIZE, rf_dim);
    Eigen::Ref<Eigen::MatrixXd> Wl_block = Wl.block(offset, 0, LINE_BLOCK_SIZE, m_dim);
    Sll_blocks_[i].noalias() += PQl_block * PQl_block.transpose() + Wl_block * Wl_block.transpose();
  }
#pragma omp parallel for shared(PQl, Wl, Sll_blocks_) 
  for(int i = 0; i < retain_surf_block_size_; i++) {
    int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + i * SURF_BLOCK_SIZE;
    Eigen::Ref<Eigen::MatrixXd> PQl_block = PQl.block(offset, 0, SURF_BLOCK_SIZE, rf_dim);
    Eigen::Ref<Eigen::MatrixXd> Wl_block = Wl.block(offset, 0, SURF_BLOCK_SIZE, m_dim);
    Sll_blocks_[i + retain_line_block_size_] += PQl_block * PQl_block.transpose() + Wl_block * Wl_block.transpose();
  }

  Slf_ = -PQ + Wl * Wf.transpose();
  Sff_ = Q + Wf * Wf.transpose();
  printf("[NFRSolver] Building sparse structure done! Time: %lf ms\n", timer.toc());
}

void NFRSolver::checkAccuracy() {
  uint64_t lm_dim = retain_line_block_size_ * LINE_BLOCK_SIZE + retain_surf_block_size_ * SURF_BLOCK_SIZE;
  uint64_t rf_dim = retain_frame_block_size_ * POSE_BLOCK_SIZE;
  Eigen::MatrixXd ratio;
  for(int i = 0; i < retain_line_block_size_; i++) {
    int offset = i * LINE_BLOCK_SIZE;
    Eigen::MatrixXd error_Sll = marg_covariance_.block(offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE) - Sll_blocks_[i];
    // std::cout << "Dense Mat: " << std::endl << std::endl << marg_covariance_.block(offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE) << std::endl;;
    ratio = error_Sll.cwiseQuotient(Sll_blocks_[i]);
    printf("error_Sll error max: %.9lf, min: %.9lf\n", ratio.maxCoeff(), ratio.minCoeff());
  }
  for(int i = 0; i < retain_surf_block_size_; i++) {
    int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + i * SURF_BLOCK_SIZE;
    Eigen::MatrixXd error_Sll = marg_covariance_.block(offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE) - Sll_blocks_[i + retain_line_block_size_];
    // std::cout << "Dense Mat: " << std::endl << std::endl << marg_covariance_.block(offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE) << std::endl;;
    ratio = error_Sll.cwiseQuotient(Sll_blocks_[i + retain_line_block_size_]);
    printf("error_Sll error max: %.9lf, min: %.9lf\n", ratio.maxCoeff(), ratio.minCoeff());
  } 

  Eigen::MatrixXd error_Slf = marg_covariance_.block(0, lm_dim, lm_dim, rf_dim) - Slf_;
  // std::cout << "Dense Mat: " << std::endl << std::endl << marg_covariance_.block(0, lm_dim, lm_dim, rf_dim) << std::endl;;
  // std::cout << "Sparse Mat: " << std::endl << Slf_ << std::endl;
  ratio = error_Slf.cwiseQuotient(Slf_);
  printf("error_Slf error max: %.9lf, min: %.9lf\n", ratio.maxCoeff(), ratio.minCoeff());

  Eigen::MatrixXd error_Sff = marg_covariance_.block(lm_dim, lm_dim, rf_dim, rf_dim) - Sff_;
  // std::cout << "Dense Mat: " << std::endl << std::endl << marg_covariance_.block(lm_dim, lm_dim, rf_dim, rf_dim) << std::endl;;
  // std::cout << "Sparse Mat: " << std::endl << Sff_ << std::endl;
  ratio = error_Sff.cwiseQuotient(Sff_);
  printf("error_Sff error max: %.9lf, min: %.9lf\n", ratio.maxCoeff(), ratio.minCoeff());
}

// void NFRSolver::buildSparseStructCUDA() {
//   printf("[NFRSolver] Start to build sparse structure!\n");
//   uint64_t total_size = marg_parameter_size_ + retain_parameter_size_;
//   std::map<uint64_t, std::map<uint64_t, std::vector<Eigen::MatrixXd>>> hessian_map; // row block id, col block id, Hkl
//   uint64_t lm_block_size = retain_line_block_size_ + retain_surf_block_size_;
//   uint64_t retain_block_size = retain_line_block_size_ + retain_surf_block_size_ + retain_frame_block_size_;
//   uint64_t total_block_size = retain_line_block_size_ + retain_surf_block_size_ + retain_frame_block_size_ + marg_block_size_;

//   uint64_t lm_dim = retain_line_block_size_ * LINE_BLOCK_SIZE + retain_surf_block_size_ * SURF_BLOCK_SIZE;
//   uint64_t rf_dim = retain_frame_block_size_ * POSE_BLOCK_SIZE;
//   uint64_t r_dim = lm_dim + rf_dim;
//   uint64_t m_dim = marg_block_size_ * POSE_BLOCK_SIZE;

//   A_.resize(lm_dim, lm_dim);
//   U_.resize(lm_dim, rf_dim);
//   C_.resize(rf_dim, rf_dim);
//   Vl_.resize(lm_dim, m_dim);
//   Vf_.resize(rf_dim, m_dim);
//   Vm_.resize(m_dim, m_dim);

//   printf("[NFRSolver] Evaluate full hessian ...\n");
  
//   ceres::LossFunction *huber_loss = new ceres::HuberLoss(1.0);
//   printf("[NFRSolver] Evaluate relative pose factors ...\n");

//   omp_set_num_threads(8);
// // #pragma omp parallel for shared(rel_pose_info_, hessian_map) 
//   for(size_t i = 0; i < rel_pose_info_.size(); i++) {
//     RelPoseInfo& info = rel_pose_info_[i];
//     OrderingType fibdim, fjbdim;
//     if(isMarginalized(info.fi)) {
//       fibdim = getMargFrameBlockDim(info.fi);
//     }
//     else {
//       fibdim = getRetainFrameBlockDim(info.fi);
//     }
//     if(isMarginalized(info.fj)) {
//       fjbdim = getMargFrameBlockDim(info.fj);
//     }
//     else {
//       fjbdim = getRetainFrameBlockDim(info.fj);
//     }

//     RelativePoseFactor *factor = new RelativePoseFactor(info.Tij, info.sqrt_info);
//     std::vector<double*> parameter_blocks{info.fi->Twb().parameters().data(), info.fj->Twb().parameters().data()};
//     std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
//     Eigen::VectorXd residuals;
//     evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

//     Eigen::MatrixXd Hii, Hij, Hjj;
//     Hii = (jacobians[0].transpose() * jacobians[0]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
//     Hjj = (jacobians[1].transpose() * jacobians[1]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
//     Hij = (jacobians[0].transpose() * jacobians[1]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);

//     hessian_map[fibdim][fibdim].push_back(Hii);
//     hessian_map[fjbdim][fjbdim].push_back(Hjj);
//     if(fibdim < fjbdim) {
//       hessian_map[fibdim][fjbdim].push_back(Hij);      
//     }
//     else if(fibdim > fjbdim) {
//       hessian_map[fjbdim][fibdim].push_back(Hij.transpose());  
//     }
//     else {
//       printf("[NFRSolver] Warning! There is an relative pose measurement between two same frames!\n");
//     }

//   }
//   printf("[NFRSolver] Evaluate line factors ...\n");

//   std::vector<LineLM::Ptr> retain_line_lms;
//   for (auto const& element : map_retain_line_) {
//     retain_line_lms.push_back(element.first);
//   }
// #pragma omp parallel for shared(retain_line_lms, hessian_map)   
//   for(int i = 0; i < retain_line_lms.size(); ++i) {
//     const LineLM::Ptr lm = retain_line_lms[i];
//     const OrderingType ldim = getLineDim(lm);
//     const OrderingType lbdim = getLineBlockDim(lm);

//     omp_set_lock(&lock);
//     auto obs = lm->getAllObs();
//     for(auto &ob: obs) {
//       Frame::Ptr frame = ob.first;
//       OrderingType fdim, fbdim, fid;
//       if(isMarginalized(frame)) {
//         fdim = getMargFrameDim(frame);
//         fbdim = getMargFrameBlockDim(frame);
//       }
//       else {
//         fdim = getRetainFrameDim(frame);
//         fbdim = getRetainFrameBlockDim(frame);
//       }
//       LaserEdge2PFactor *factor = new LaserEdge2PFactor(ob.second->point_a(), ob.second->point_b(), ob.second->sqrt_info());
//       std::vector<double*> parameter_blocks{frame->Twb().parameters().data(), lm->parameters().data()};
//       std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
//       Eigen::VectorXd residuals;
//       evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

//       Eigen::MatrixXd Hpp, Hll, Hlp;
//       Hpp = (jacobians[0].transpose() * jacobians[0]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
//       Hll = (jacobians[1].transpose() * jacobians[1]);
//       Hlp = (jacobians[1].transpose() * jacobians[0]).leftCols<POSE_BLOCK_SIZE>();

//       hessian_map[lbdim][lbdim].push_back(Hll);
//       hessian_map[fbdim][fbdim].push_back(Hpp);
//       hessian_map[lbdim][fbdim].push_back(Hlp);
//     }
//     omp_unset_lock(&lock);
//   }

//   printf("[NFRSolver] Evaluate surf factors ...\n");
//   std::vector<SurfaceLM::Ptr> retain_surf_lms;
//   for (auto const& element : map_retain_surf_) {
//     retain_surf_lms.push_back(element.first);
//   }

// #pragma omp parallel for shared(retain_surf_lms, hessian_map) 
//   for(int i = 0; i < retain_surf_lms.size(); ++i) {
//     const SurfaceLM::Ptr lm = retain_surf_lms[i];
//     const OrderingType sdim = getSurfDim(lm);
//     const OrderingType sbdim = getSurfBlockDim(lm);
//     auto obs = lm->getAllObs();

//     omp_set_lock(&lock);
//     for(auto &ob: obs) {
//       Frame::Ptr frame = ob.first;
//       OrderingType fdim, fbdim, fid;
//       if(isMarginalized(frame)) {
//         fdim = getMargFrameDim(frame);
//         fbdim = getMargFrameBlockDim(frame);
//       }
//       else {
//         fdim = getRetainFrameDim(frame);
//         fbdim = getRetainFrameBlockDim(frame);
//       }
 
//       SurfaceOB::Ptr feature = ob.second;
//       std::vector<Eigen::Vector3d> const vertices = feature->vertices();
//       // LaserSurf4PFactor *factor = new LaserSurf4PFactor(vertices, feature->sqrt_info());
//       LaserSurf3PFactor *factor = new LaserSurf3PFactor(vertices, feature->sqrt_info());
//       std::vector<double*> parameter_blocks{frame->Twb().parameters().data(), lm->parameters().data()};
//       std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
//       Eigen::VectorXd residuals;
//       evalFactor(factor, huber_loss, parameter_blocks, residuals, jacobians);

//       Eigen::MatrixXd Hpp, Hll, Hlp;
//       Hpp = (jacobians[0].transpose() * jacobians[0]).block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(0, 0);
//       Hll = (jacobians[1].transpose() * jacobians[1]);
//       Hlp = (jacobians[1].transpose() * jacobians[0]).leftCols<POSE_BLOCK_SIZE>();

//       hessian_map[sbdim][sbdim].push_back(Hll);
//       hessian_map[fbdim][fbdim].push_back(Hpp);
//       hessian_map[sbdim][fbdim].push_back(Hlp);
//     }
//     omp_unset_lock(&lock);
//   }

//   for(auto fixed_frame: fixed_frames_) {
//     auto fbdim = getRetainFrameBlockDim(fixed_frame);
//     hessian_map[fbdim][fbdim].push_back(Eigen::MatrixXd::Identity(POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) * 1e8);
//   }

//   printf("[NFRSolver] Fill in the sparse blocks ...\n");
//   A_blocks_.clear();
//   std::vector<Eigen::Triplet<double>> U_trip, C_trip, Vl_trip, Vf_trip, Vm_trip;
// // #pragma omp parallel for shared(A_blocks_, C_, U_, Vl_, Vf_, Vm_, hessian_map) 
//   for(auto row_iter = hessian_map.begin(); row_iter != hessian_map.end(); row_iter++) {
//     int row_block_id = row_iter->first;
//     for(auto col_iter = row_iter->second.begin(); col_iter != row_iter->second.end(); col_iter++) {
//       int col_block_id = col_iter->first;
//       if(row_block_id < lm_block_size && col_block_id < lm_block_size) { // A block
//         for(int i = 1; i < col_iter->second.size(); i++) {
//           col_iter->second.front().noalias() += col_iter->second[i];
//         }
//         A_blocks_.push_back(col_iter->second.front());
//       }
//       else if(row_block_id < lm_block_size && col_block_id >= lm_block_size && col_block_id < retain_block_size) { // U block
//         for(int i = 1; i < col_iter->second.size(); i++) {
//           col_iter->second.front().noalias() += col_iter->second[i];
//         }
//         Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
//         if(row_block_id < retain_line_block_size_) {
//           int row_offset = row_block_id * LINE_BLOCK_SIZE;
//           int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
//           for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
//             for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//               U_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
//             }
//           }
//           // writeBlockRowMajor(Ur_, col_iter->second.front(), row_offset, col_offset, LINE_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         }
//         else {
//           int row_offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (row_block_id - retain_line_block_size_) * SURF_BLOCK_SIZE;
//           int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
//           for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
//             for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//               U_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
//             }
//           }
//           // writeBlockRowMajor(Ur_, col_iter->second.front(), row_offset, col_offset, SURF_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         }
//       }
//       else if(row_block_id >= lm_block_size && row_block_id < retain_block_size && col_block_id >= lm_block_size && col_block_id < retain_block_size) { // C block
//         for(int i = 1; i < col_iter->second.size(); i++) {
//           col_iter->second.front().noalias() += col_iter->second[i];
//         }
//         Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
//         int row_offset = (row_block_id - lm_block_size) * POSE_BLOCK_SIZE;
//         int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
//         for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
//           for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//             C_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
//           }
//         }
//         if(row_block_id != col_block_id) {
//           for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
//             for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//               C_trip.push_back(Eigen::Triplet<double>(col_offset + i, row_offset + j, block(j, i)));
//             }
//           }          
//         }
//         // writeBlockRowMajor(Cr_, col_iter->second.front(), row_offset, col_offset, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         // if(row_block_id != col_block_id) {
//         //   writeBlockRowMajor(Cr_, col_iter->second.front().transpose(), col_offset, row_offset, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         // }
//       }
//       else if(row_block_id < lm_block_size && col_block_id >= retain_block_size && col_block_id < total_block_size) { // Vl block
//         for(int i = 1; i < col_iter->second.size(); i++) {
//           col_iter->second.front().noalias() += col_iter->second[i];
//         }
//         Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
//         if(row_block_id < retain_line_block_size_) {
//           int row_offset = row_block_id * LINE_BLOCK_SIZE;
//           int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
//           for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
//             for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//               Vl_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
//             }
//           }
//           // writeBlockRowMajor(Vlr_, col_iter->second.front(), row_offset, col_offset, LINE_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         }
//         else {
//           int row_offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (row_block_id - retain_line_block_size_) * SURF_BLOCK_SIZE;
//           int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
//           for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
//             for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//               Vl_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
//             }
//           }
//           // writeBlockRowMajor(Vlr_, col_iter->second.front(), row_offset, col_offset, SURF_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         }
//       }
//       else if(row_block_id >= lm_block_size && row_block_id < retain_block_size && col_block_id >= retain_block_size && col_block_id < total_block_size) { // Vf block
//         for(int i = 1; i < col_iter->second.size(); i++) {
//           col_iter->second.front().noalias() += col_iter->second[i];
//         }
//         Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
//         int row_offset = (row_block_id - lm_block_size) * POSE_BLOCK_SIZE;
//         int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
//         for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
//           for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//             Vf_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
//           }
//         }
//         // writeBlockRowMajor(Vfr_, col_iter->second.front(), row_offset, col_offset, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE);
//       }
//       else if(row_block_id >= retain_block_size && row_block_id < total_block_size && col_block_id >= retain_block_size && col_block_id < total_block_size) { // Hmm block
//         for(int i = 1; i < col_iter->second.size(); i++) {
//           col_iter->second.front().noalias() += col_iter->second[i];
//         }
//         Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
//         int row_offset = (row_block_id - retain_block_size) * POSE_BLOCK_SIZE;
//         int col_offset = (col_block_id - retain_block_size) * POSE_BLOCK_SIZE;
//         for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
//           for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//             Vm_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
//           }
//         }
//         if(row_block_id != col_block_id) {
//           for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
//             for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
//               Vm_trip.push_back(Eigen::Triplet<double>(col_offset + i, row_offset + j, block(j, i)));
//             }
//           }
//         }
//         // writeBlockRowMajor(Vmr_, col_iter->second.front(), row_offset, col_offset, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         // if(row_block_id != col_block_id) {
//         //   writeBlockRowMajor(Vmr_, col_iter->second.front().transpose(), col_offset, row_offset, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE);
//         // }
//       }
//       else {
//         printf("[NFRSolver] Warning! There is an error index matrix in NFR! (row_block_id: %d, col_block_id: %d)\n", row_block_id, col_block_id);
//       }
//     }
//   }

//   U_.setFromTriplets(U_trip.begin(), U_trip.end());
//   C_.setFromTriplets(C_trip.begin(), C_trip.end());
//   Vl_.setFromTriplets(Vl_trip.begin(), Vl_trip.end());
//   Vf_.setFromTriplets(Vf_trip.begin(), Vf_trip.end());
//   Vm_.setFromTriplets(Vm_trip.begin(), Vm_trip.end());

//   U_.makeCompressed();
//   C_.makeCompressed();
//   Vl_.makeCompressed();
//   Vf_.makeCompressed();
//   Vm_.makeCompressed();

//   // printf("[NFRSolver] Write diagonal blocks ...\n");
//   // for(int i = 0; i < retain_line_block_size_; i++) {
//   //   int offset = i * LINE_BLOCK_SIZE;
//   //   writeBlock(A_, A_blocks_[i], offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE);
//   // }
//   // for(int i = 0; i < retain_surf_block_size_; i++) {
//   //   int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + i * SURF_BLOCK_SIZE;
//   //   writeBlock(A_, A_blocks_[i + retain_line_block_size_], offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE);
//   // }

//   // A_.makeCompressed();

//   // Eigen::MatrixXd error;
//   // Eigen::MatrixXd dense_A = A_.toDense();
//   // error = dense_A - hessian_.block(0, 0, lm_dim, lm_dim);
//   // printf("error A: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());
  
//   // Eigen::MatrixXd dense_C = C_.toDense();
//   // error = dense_C - hessian_.block(lm_dim, lm_dim, rf_dim, rf_dim);
//   // printf("error C: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

//   // Eigen::MatrixXd dense_Vm = Vm_.toDense();
//   // error = dense_Vm - hessian_.block(lm_dim + rf_dim, lm_dim + rf_dim, m_dim, m_dim);
//   // printf("error Vm: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

//   // Eigen::MatrixXd dense_U = U_.toDense();
//   // error = dense_U - hessian_.block(0, lm_dim, lm_dim, rf_dim);
//   // printf("error U: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

//   // Eigen::MatrixXd dense_Vl = Vl_.toDense();
//   // error = dense_Vl - hessian_.block(0, r_dim, lm_dim, m_dim);
//   // printf("error Vl: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

//   // Eigen::MatrixXd dense_Vf = Vf_.toDense();
//   // error = dense_Vf - hessian_.block(lm_dim, r_dim, rf_dim, m_dim);
//   // printf("error Vf: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

  
//   TicToc timer;
//   printf("[NFRSolver] Compute block-wise inverse ...\n");
//   Ainv_.resize(lm_dim, lm_dim);
//   Sll_blocks_.resize(A_blocks_.size());
// #pragma omp parallel for shared(A_blocks_, Sll_blocks_) 
//   for(int i = 0; i < A_blocks_.size(); i++) {
//     if(i < retain_line_block_size_) {
//       int offset = i * LINE_BLOCK_SIZE;
//       Eigen::MatrixXd inv_block = A_blocks_[i].inverse();
//       Sll_blocks_[i] = inv_block;
//       // writeBlock(Ainv_, inv_block, offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE);
//     }
//     else {
//       int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (i - retain_line_block_size_) * SURF_BLOCK_SIZE;
//       Eigen::MatrixXd inv_block = A_blocks_[i].inverse();
//       Sll_blocks_[i] = inv_block;
//       // writeBlock(Ainv_, inv_block, offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE);
//     }
//   }

//   for(int i = 0; i < A_blocks_.size(); i++) {
//     if(i < retain_line_block_size_) {
//       int offset = i * LINE_BLOCK_SIZE;
//       writeBlock(Ainv_, Sll_blocks_[i], offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE);
//     }
//     else {
//       int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (i - retain_line_block_size_) * SURF_BLOCK_SIZE;
//       writeBlock(Ainv_, Sll_blocks_[i], offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE);
//     }
//   }

//   Ainv_.makeCompressed();

//   printf("[NFRSolver] Evaluate sparse covariance ...\n");
//   timer.tic();

//   // CuSMat dC(Cr_), dUt(Ur_.transpose()), dAinv(Ainvr_), dU(Ur_), dVl(Vlr_);
//   // CuDMat dVf(Vfr_.toDense());
//   // CuSMat dP = dAinv * dU;
//   // CuSMat dPt = dUt * dAinv;
//   // CuSMat dQinv = dC - (dUt * dP);
//   // CuDMat dQinvd(dQinv.toEigenSparse().toDense());
//   // Eigen::MatrixXd Q = dQinvd.inverse().toEigenDense();
//   // Eigen::MatrixXd Ql = solveCholeskyCUDA(Q);
//   // CuDMat dPQl = dP * CuDMat(dQl);
//   // CuSMat dPtVl = dPt * dVl;
//   // CuSMat dDVl = dAinv * dVl;
//   // CuDMat dQ(Q);
//   // CuDMat dQVf = dQ * dVf;
//   // CuDMat dPQVf = dP * dQVf;
//   // std::cout << "cuda basic factors time: " << timer.toc() << std::endl;

//   // std::cout << "cusparse Q: " << std::endl << dQ << std::endl;

//   Eigen::SparseMatrix<double> Qinv = Cr_ - Ur_.transpose() * Ainvr_ * Ur_;
//   Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> Qqr(Qinv);
//   Eigen::MatrixXd Q = Qqr.solve(Eigen::MatrixXd::Identity(Qinv.rows(), Qinv.cols()));
//   Eigen::SparseMatrix<double> P = Ainvr_ * Ur_;

//   // // compute basic factors
//   timer.tic();
//   Eigen::MatrixXd Ql = Eigen::LLT<Eigen::MatrixXd>(Q).matrixL();
//   Eigen::MatrixXd PQl = P * Ql;                             // P * Ql            [lxf]


//   Eigen::SparseMatrix<double> PtVl = P.transpose() * Vlr_;  // P^T * Vl          [fxm]
//   Eigen::SparseMatrix<double> DVl = Ainvr_ * Vlr_;          // A^{-1} * Vl       [lxm]
//   Eigen::MatrixXd QVf = Q * Vfr_;                           // Q * Vf            [fxm]
//   Eigen::MatrixXd PQVf = P * QVf;                           // P * Q * Vf        [lxm]
//   Eigen::MatrixXd QPtVl = Q * PtVl;                         // Q * P^T * Vl      [fxm]
//   Eigen::MatrixXd PQPtVl = P * QPtVl;                       // P * Q * P^T * Vl  [lxm]
//   std::cout << "basic factors time: " << timer.toc() << std::endl;

//   // error = dQl - Ql;
//   // printf("Ql error: max: %lf, min: %lf\n", error.maxCoeff(), error.minCoeff());

//   // calculate marg Schur
//   timer.tic();
//   Eigen::MatrixXd VltPQVf = Vl_.transpose() * PQVf; // [mxl] x [lxm] = [mxm]
//   Eigen::MatrixXd Hm_schur_cross = VltPQVf + VltPQVf.transpose();
//   Eigen::MatrixXd Rinv = Vm_ - (Vl_.transpose() * DVl + PtVl.transpose() * Q * PtVl + Vf_.transpose() * QVf - Hm_schur_cross);
//   Eigen::MatrixXd R = Rinv.inverse();
//   Eigen::MatrixXd Rl = Eigen::LLT<Eigen::MatrixXd>(R).matrixL();  

//   Eigen::MatrixXd PtVlmVfRl = (PtVl - Vf_) * Rl;
//   Eigen::MatrixXd PQ = P * Q;
//   Eigen::MatrixXd Wl = Ainv_ * (Vl_ * Rl) + PQ * PtVlmVfRl; // [lxm]
//   Eigen::MatrixXd Wf = -Q * PtVlmVfRl;        // [fxm]
//   std::cout << "second basic factors time: " << timer.toc() << std::endl;

//   timer.tic();
// #pragma omp parallel for shared(PQl, Wl, Sll_blocks_) 
//   for(int i = 0; i < retain_line_block_size_; i++) {
//     int offset = i * LINE_BLOCK_SIZE;
//     Eigen::Ref<Eigen::MatrixXd> PQl_block = PQl.block(offset, 0, LINE_BLOCK_SIZE, rf_dim);
//     Eigen::Ref<Eigen::MatrixXd> Wl_block = Wl.block(offset, 0, LINE_BLOCK_SIZE, m_dim);
//     Sll_blocks_[i].noalias() += PQl_block * PQl_block.transpose() + Wl_block * Wl_block.transpose();
//   }
// #pragma omp parallel for shared(PQl, Wl, Sll_blocks_) 
//   for(int i = 0; i < retain_surf_block_size_; i++) {
//     int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + i * SURF_BLOCK_SIZE;
//     Eigen::Ref<Eigen::MatrixXd> PQl_block = PQl.block(offset, 0, SURF_BLOCK_SIZE, rf_dim);
//     Eigen::Ref<Eigen::MatrixXd> Wl_block = Wl.block(offset, 0, SURF_BLOCK_SIZE, m_dim);
//     Sll_blocks_[i + retain_line_block_size_] += PQl_block * PQl_block.transpose() + Wl_block * Wl_block.transpose();
//   }

//   Slf_ = -PQ + Wl * Wf.transpose();
//   Sff_ = Q + Wf * Wf.transpose();
//   std::cout << "final factors time: " << timer.toc() << std::endl;

//   printf("[NFRSolver] Building sparse structure done! Time: %lf ms\n", timer.toc());
// }

void NFRSolver::addFrameToLineNF(const Frame::Ptr& f, const LineLM::Ptr& lm) {
  uint64_t ldim = getLineDim(lm);
  uint64_t lbdim = getLineBlockDim(lm);
  uint64_t fdim = getRetainFrameDim(f) - getSurfDimOffset();

  int point_num;
  std::vector<Eigen::Vector3d> virtual_obs;
  Eigen::MatrixXd Jf, Jl;
  evalFrameToLine(f, lm, Jf, Jl, virtual_obs, point_num);
  FrameToLineNF factor;
  factor.frame_ptr = f;
  factor.lm_ptr = lm;
  factor.Jf = Jf;
  factor.Jl = Jl;
  factor.ob = virtual_obs;
  factor.point_num = point_num;
  line_factors_.push_back(factor);

  // std::cout << "p1: " << virtual_obs[0].transpose() << std::endl;
  // std::cout << "p2: " << virtual_obs[1].transpose() << std::endl;
  // std::cout << "nums: " << point_num << std::endl;
  // std::cout << "Jf: " << std::endl << Jf << std::endl;
  // std::cout << "Jl: " << std::endl << Jl << std::endl;
}

void NFRSolver::addFrameToSurfNF(const Frame::Ptr& f, const SurfaceLM::Ptr& lm) {
  uint64_t sdim = getSurfDim(lm);
  uint64_t sbdim = getSurfBlockDim(lm);
  uint64_t fdim = getRetainFrameDim(f) - getSurfDimOffset();

  int point_num;
  std::vector<Eigen::Vector3d> virtual_obs;
  Eigen::MatrixXd Jf, Jl;
  evalFrameToSurf(f, lm, Jf, Jl, virtual_obs, point_num);
  FrameToSurfNF factor;
  factor.frame_ptr = f;
  factor.lm_ptr = lm;
  factor.Jf = Jf;
  factor.Jl = Jl;
  factor.ob = virtual_obs;
  factor.point_num = point_num;
  surf_factors_.push_back(factor);
}

void NFRSolver::addFrameToFrameNF(const Frame::Ptr& fi, const Frame::Ptr& fj) {
  uint64_t fidim = getRetainFrameDim(fi) - getSurfDimOffset();
  uint64_t fjdim = getRetainFrameDim(fj) - getSurfDimOffset();

  Transform virtual_ob;
  Eigen::MatrixXd Jfi, Jfj;
  evalFrameToFrame(fi, fj, Jfi, Jfj, virtual_ob);
  FrameToFrameNF factor;
  factor.fi_ptr = fi;
  factor.fj_ptr = fj;
  factor.Jfi = Jfi;
  factor.Jfj = Jfj;
  factor.ob = virtual_ob;
  rel_pose_factors_.push_back(factor);
}

void NFRSolver::addFramePriorNF(const Frame::Ptr& frame) {
  uint64_t fdim = getRetainFrameDim(frame);

  Transform virtual_ob;
  Eigen::MatrixXd Jf;
  evalFramePrior(frame, Jf, virtual_ob);
  FramePriorNF factor;
  factor.frame_ptr = frame;
  factor.Jf = Jf;
  factor.ob = virtual_ob;
  prior_factors_.push_back(factor);
}


void NFRSolver::buildTargetSigma(std::vector<Eigen::MatrixXd>& rSll_blocks, Eigen::MatrixXd& rSlf, Eigen::MatrixXd& rSff) {
  int line_vm_size = line_factors_.size();
  int surf_vm_size = surf_factors_.size();
  int rel_pose_vm_size = rel_pose_factors_.size();
  int prior_vm_size = prior_factors_.size();

  Eigen::MatrixXd rHessian(retain_parameter_size_, retain_parameter_size_);
  rHessian.setZero();
  std::map<uint64_t, std::map<uint64_t, std::vector<Eigen::MatrixXd>>> hessian_map; // row block id, col block id, Hkl

  // for(auto fixed_frame: fixed_frames_) {
  //   auto fbdim = getRetainFrameBlockDim(fixed_frame);
  //   auto fdim = getRetainFrameDim(fixed_frame, 0);
  //   Eigen::MatrixXd Jf = line_factors_[index].sqrt_info * line_factors_[index].Jf;
  //   Eigen::MatrixXd Jl = line_factors_[index].sqrt_info * line_factors_[index].Jl;

  //   rHessian.block<6, 6>(fdim, fdim).noalias() += Eigen::Matrix<double, 6, 6>::Identity() * 1e8;
  //   hessian_map[fbdim][fbdim].push_back(Eigen::MatrixXd::Identity(POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) * 1e8);
  // }

  for(int index = 0; index < prior_vm_size; ++index) {
    auto fbdim = getRetainFrameBlockDim(prior_factors_[index].frame_ptr);
    auto fdim = getRetainFrameDim(prior_factors_[index].frame_ptr, 0);
    Eigen::MatrixXd Jf = prior_factors_[index].sqrt_info * prior_factors_[index].Jf;
    rHessian.block<6, 6>(fdim, fdim).noalias() += Jf.transpose() * Jf;
    hessian_map[fbdim][fbdim].push_back(Jf.transpose() * Jf);
  }

  for(int index = 0; index < line_vm_size; ++index) {
    uint64_t fdim = getRetainFrameDim(line_factors_[index].frame_ptr);
    uint64_t fbdim = getRetainFrameBlockDim(line_factors_[index].frame_ptr);
    uint64_t ldim = getLineDim(line_factors_[index].lm_ptr);
    uint64_t lbdim = getLineBlockDim(line_factors_[index].lm_ptr); 
    Eigen::MatrixXd Jf = line_factors_[index].sqrt_info * line_factors_[index].Jf;
    Eigen::MatrixXd Jl = line_factors_[index].sqrt_info * line_factors_[index].Jl;
    rHessian.block<LINE_BLOCK_SIZE, LINE_BLOCK_SIZE>(ldim, ldim) += Jl.transpose() * Jl;
    rHessian.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim) += Jf.transpose() * Jf;
    rHessian.block<LINE_BLOCK_SIZE, POSE_BLOCK_SIZE>(ldim, fdim) += Jl.transpose() * Jf;
    rHessian.block<POSE_BLOCK_SIZE, LINE_BLOCK_SIZE>(fdim, ldim) += Jf.transpose() * Jl;

    hessian_map[fbdim][fbdim].push_back(Jf.transpose() * Jf);
    hessian_map[lbdim][lbdim].push_back(Jl.transpose() * Jl);
    hessian_map[lbdim][fbdim].push_back(Jl.transpose() * Jf);
  }

  for(int index = 0; index < surf_vm_size; ++index) {
    uint64_t fdim = getRetainFrameDim(surf_factors_[index].frame_ptr);
    uint64_t fbdim = getRetainFrameBlockDim(surf_factors_[index].frame_ptr);
    uint64_t sdim = getSurfDim(surf_factors_[index].lm_ptr);
    uint64_t sbdim = getSurfBlockDim(surf_factors_[index].lm_ptr); 
    Eigen::MatrixXd Jf = surf_factors_[index].sqrt_info * surf_factors_[index].Jf;
    Eigen::MatrixXd Jl = surf_factors_[index].sqrt_info * surf_factors_[index].Jl;
    rHessian.block<SURF_BLOCK_SIZE, SURF_BLOCK_SIZE>(sdim, sdim) += Jl.transpose() * Jl;
    rHessian.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim) += Jf.transpose() * Jf;
    rHessian.block<SURF_BLOCK_SIZE, POSE_BLOCK_SIZE>(sdim, fdim) += Jl.transpose() * Jf;
    rHessian.block<POSE_BLOCK_SIZE, SURF_BLOCK_SIZE>(fdim, sdim) += Jf.transpose() * Jl;

    hessian_map[fbdim][fbdim].push_back(Jf.transpose() * Jf);
    hessian_map[sbdim][sbdim].push_back(Jl.transpose() * Jl);
    hessian_map[sbdim][fbdim].push_back(Jl.transpose() * Jf);
  }

  for(int index = 0; index < rel_pose_vm_size; ++index) {
    uint64_t fidim = getRetainFrameDim(rel_pose_factors_[index].fi_ptr);
    uint64_t fibdim = getRetainFrameBlockDim(rel_pose_factors_[index].fi_ptr);
    uint64_t fjdim = getRetainFrameDim(rel_pose_factors_[index].fj_ptr);
    uint64_t fjbdim = getRetainFrameBlockDim(rel_pose_factors_[index].fj_ptr);
    assert(fibdim != fjbdim);

    Eigen::MatrixXd Jfi = rel_pose_factors_[index].sqrt_info * rel_pose_factors_[index].Jfi;
    Eigen::MatrixXd Jfj = rel_pose_factors_[index].sqrt_info * rel_pose_factors_[index].Jfj;
    rHessian.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fidim) += Jfi.transpose() * Jfi;
    rHessian.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fjdim, fjdim) += Jfj.transpose() * Jfj;
    rHessian.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fjdim) += Jfi.transpose() * Jfj;
    rHessian.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fjdim, fidim) += Jfj.transpose() * Jfi;

    hessian_map[fibdim][fibdim].push_back(Jfi.transpose() * Jfi);
    hessian_map[fjbdim][fjbdim].push_back(Jfj.transpose() * Jfj);
    if(fibdim < fjbdim) {
      hessian_map[fibdim][fjbdim].push_back(Jfi.transpose() * Jfj);
    }
    else {
      hessian_map[fjbdim][fibdim].push_back(Jfj.transpose() * Jfi);
    }
  }

  uint64_t lm_dim = retain_line_block_size_ * LINE_BLOCK_SIZE + retain_surf_block_size_ * SURF_BLOCK_SIZE;
  uint64_t rf_dim = retain_frame_block_size_ * POSE_BLOCK_SIZE;
  uint64_t lm_block_size = retain_line_block_size_ + retain_surf_block_size_;
  uint64_t retain_block_size = retain_line_block_size_ + retain_surf_block_size_ + retain_frame_block_size_;

  std::vector<Eigen::MatrixXd> Hll_line_blocks, Hll_surf_blocks;
  std::vector<Eigen::Triplet<double>> Hlf_trip, Hff_trip;
  
  for(auto row_iter = hessian_map.begin(); row_iter != hessian_map.end(); row_iter++) {
    int row_block_id = row_iter->first;
    for(auto col_iter = row_iter->second.begin(); col_iter != row_iter->second.end(); col_iter++) {
      int col_block_id = col_iter->first;
      if(row_block_id < lm_block_size && col_block_id < lm_block_size) { // Hll block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        if(row_block_id < retain_line_block_size_) {
          Hll_line_blocks.push_back(col_iter->second.front());
        }
        else {
          Hll_surf_blocks.push_back(col_iter->second.front());
        }
      }
      else if(row_block_id < lm_block_size && col_block_id >= lm_block_size && col_block_id < retain_block_size) { // Hlf block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
        if(row_block_id < retain_line_block_size_) {
          int row_offset = row_block_id * LINE_BLOCK_SIZE;
          int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
          for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              Hlf_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
            }
          }
        }
        else {
          int row_offset = retain_line_block_size_ * LINE_BLOCK_SIZE + (row_block_id - retain_line_block_size_) * SURF_BLOCK_SIZE;
          int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
          for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              Hlf_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
            }
          }
        }
      }
      else if(row_block_id >= lm_block_size && row_block_id < retain_block_size && col_block_id >= lm_block_size && col_block_id < retain_block_size) { // Hff block
        for(int i = 1; i < col_iter->second.size(); i++) {
          col_iter->second.front().noalias() += col_iter->second[i];
        }
        Eigen::Ref<Eigen::MatrixXd> block = col_iter->second.front();
        int row_offset = (row_block_id - lm_block_size) * POSE_BLOCK_SIZE;
        int col_offset = (col_block_id - lm_block_size) * POSE_BLOCK_SIZE;
        for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
          for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
            Hff_trip.push_back(Eigen::Triplet<double>(row_offset + i, col_offset + j, block(i, j)));
          }
        }
        if(row_block_id != col_block_id) {
          for(int i = 0; i < POSE_BLOCK_SIZE; i++) {
            for(int j = 0; j < POSE_BLOCK_SIZE; j++) {
              Hff_trip.push_back(Eigen::Triplet<double>(col_offset + i, row_offset + j, block(j, i)));
            }
          }          
        }
      }
      else {
        printf("[NFRSolver] Warning! There is an error in dimension!\n");
      }
    }
  }
  
  Eigen::SparseMatrix<double> Hll, Hllinv, Hlf, Hff;
  Hll.resize(lm_dim, lm_dim);
  Hllinv.resize(lm_dim, lm_dim);
  Hlf.resize(lm_dim, rf_dim);
  Hff.resize(rf_dim, rf_dim);

  Hlf.setFromTriplets(Hlf_trip.begin(), Hlf_trip.end());
  Hff.setFromTriplets(Hff_trip.begin(), Hff_trip.end());
  Hlf.makeCompressed();
  Hff.makeCompressed();

  std::vector<Eigen::Triplet<double>> Hll_trip;
  for(int index = 0; index < Hll_line_blocks.size(); index++) {
    int offset = index * LINE_BLOCK_SIZE;
    for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
      for(int j = 0; j < LINE_BLOCK_SIZE; j++) {
        Hll_trip.push_back(Eigen::Triplet<double>(offset + i, offset + j, Hll_line_blocks[index](i, j)));
      }
    }
  }
  for(int index = 0; index < Hll_surf_blocks.size(); index++) {
    int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + index * SURF_BLOCK_SIZE;
    for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
      for(int j = 0; j < SURF_BLOCK_SIZE; j++) {
        Hll_trip.push_back(Eigen::Triplet<double>(offset + i, offset + j, Hll_surf_blocks[index](i, j)));
      }
    }
  }
  Hll.setFromTriplets(Hll_trip.begin(), Hll_trip.end());
  Hll.makeCompressed();


  // gpu parallel
  std::vector<Eigen::MatrixXd> rSll_line_blocks, rSll_surf_blocks;
  if(Hll_line_blocks.size() > 0) {
    rSll_line_blocks = inverseParallelCUDA(Hll_line_blocks, 4);
  }
  if(Hll_surf_blocks.size() > 0) {
    rSll_surf_blocks = inverseParallelCUDA(Hll_surf_blocks, 3);
  }

  std::vector<Eigen::Triplet<double>> Hllinv_trip;
  for(int index = 0; index < rSll_line_blocks.size(); index++) {
    int offset = index * LINE_BLOCK_SIZE;
    for(int i = 0; i < LINE_BLOCK_SIZE; i++) {
      for(int j = 0; j < LINE_BLOCK_SIZE; j++) {
        Hllinv_trip.push_back(Eigen::Triplet<double>(offset + i, offset + j, rSll_line_blocks[index](i, j)));
      }
    }
  }
  for(int index = 0; index < rSll_surf_blocks.size(); index++) {
    int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + index * SURF_BLOCK_SIZE;
    for(int i = 0; i < SURF_BLOCK_SIZE; i++) {
      for(int j = 0; j < SURF_BLOCK_SIZE; j++) {
        Hllinv_trip.push_back(Eigen::Triplet<double>(offset + i, offset + j, rSll_surf_blocks[index](i, j)));
      }
    }
  }

  rSll_blocks.clear();
  rSll_blocks.insert(rSll_blocks.end(), rSll_line_blocks.begin(), rSll_line_blocks.end());
  rSll_blocks.insert(rSll_blocks.end(), rSll_surf_blocks.begin(), rSll_surf_blocks.end());

  Hllinv.setFromTriplets(Hllinv_trip.begin(), Hllinv_trip.end());
  Hllinv.makeCompressed();

  Eigen::MatrixXd rSigma = rHessian.inverse();

  Eigen::MatrixXd error;
  std::cout << "rHessian: " << std::endl << rHessian << std::endl;
  std::cout << "margHessian: " << std::endl << marg_hessian_ << std::endl;
  // std::cout << "rHll: " << std::endl << Hll.toDense() << std::endl;

  // std::cout << "rHessianlf: " << std::endl << rHessian.block(0, lm_dim, lm_dim, rf_dim) << std::endl;
  // std::cout << "rHlf: " << std::endl << Hlf.toDense() << std::endl;
  // std::cout << "rHessianff: " << std::endl << rHessian.block(lm_dim, lm_dim, rf_dim, rf_dim) << std::endl;
  // std::cout << "rHff: " << std::endl << Hff.toDense() << std::endl;
  // std::cout << rHessian.block(0, 0, lm_dim, lm_dim) * Hllinv.toDense() << std::endl;
  // printf("invHll error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());

  Eigen::SparseMatrix<double> rSffinv = Hff - Hlf.transpose() * Hllinv * Hlf;
  // Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> Sffqr(rSffinv);
  // rSff = Sffqr.solve(Eigen::MatrixXd::Identity(rSffinv.rows(), rSffinv.cols()));
  // Eigen::MatrixXd dSff = (Hff.toDense() - Hlf.toDense().transpose() * Hllinv.toDense() * Hlf.toDense()).inverse();

  rSff = CuDMat(rSffinv.toDense()).inverse().toEigenDense();

  // error = rSigma.block(lm_dim, lm_dim, rf_dim, rf_dim) - dSff;
  // printf("dSff error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());
  error = rSigma.block(lm_dim, lm_dim, rf_dim, rf_dim) - rSff;
  printf("rSff error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());

  // rSll = Hllinv + W * W.t, where W = Hllinv * Hlf * rSffl [lxf]
  std::vector<Eigen::MatrixXd> Sll_blocks;
  Eigen::MatrixXd rSffl = Eigen::LLT<Eigen::MatrixXd>(rSff).matrixL();  
  Eigen::MatrixXd Wl = Hllinv * Hlf * rSffl;

#pragma omp parallel for shared(Wl, Sll_blocks_) 
  for(int i = 0; i < retain_line_block_size_; i++) {
    int offset = i * LINE_BLOCK_SIZE;
    Eigen::Ref<Eigen::MatrixXd> Wl_block = Wl.block(offset, 0, LINE_BLOCK_SIZE, rf_dim);
    rSll_blocks[i].noalias() += Wl_block * Wl_block.transpose();
  }
#pragma omp parallel for shared(Wl, rSll_blocks) 
  for(int i = 0; i < retain_surf_block_size_; i++) {
    int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + i * SURF_BLOCK_SIZE;
    Eigen::Ref<Eigen::MatrixXd> Wl_block = Wl.block(offset, 0, SURF_BLOCK_SIZE, rf_dim);
    rSll_blocks[i + retain_line_block_size_] += Wl_block * Wl_block.transpose();
  }

  rSlf = -Hllinv * Hlf * rSff;
  // std::cout << "margSigma" << std::endl << marg_covariance_ << std::endl;
  // std::cout << "rSigma: " << std::endl << rSigma << std::endl;

  for(int i = 0; i < retain_line_block_size_; i++) {
    int offset = i * LINE_BLOCK_SIZE;
    error = rSll_blocks[i] - rSigma.block(offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE);
    // std::cout << "rSigmall: " << std::endl << rSigma.block(offset, offset, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE) << std::endl;
    // std::cout << "rSll: " << std::endl << rSll_blocks[i] << std::endl;
    printf("rSll error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());
  }
  for(int i = 0; i < retain_surf_block_size_; i++) {
    int offset = retain_line_block_size_ * LINE_BLOCK_SIZE + i * SURF_BLOCK_SIZE;
    error = rSll_blocks[i + retain_line_block_size_] - rSigma.block(offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE);
    // std::cout << "rSigmall: " << std::endl << rSigma.block(offset, offset, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE) << std::endl;
    // std::cout << "rSll: " << std::endl << rSll_blocks[i] << std::endl;
    printf("rSll error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());
  }

  error = rSlf - rSigma.block(0, lm_dim, lm_dim, rf_dim);
  // std::cout << "rSigmalf: " << std::endl << rSigma.block(0, lm_dim, lm_dim, rf_dim) << std::endl;
  // std::cout << "rSlf: " << std::endl << rSlf << std::endl;
  printf("rSlf error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());

  error = rSff - rSigma.block(lm_dim, lm_dim, rf_dim, rf_dim);
  // std::cout << "rSigmaff: " << std::endl << rSigma.block(lm_dim, lm_dim, rf_dim, rf_dim) << std::endl;
  // std::cout << "rSff: " << std::endl << rSff << std::endl;
  printf("rSff error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());

  // std::cout << "Error Hessian: " << std::endl << rHessian - marg_hessian_ << std::endl;

  // Eigen::MatrixXd dense_Hll = Hll.toDense();
  // Eigen::MatrixXd dense_Hlf = Hlf.toDense();
  // Eigen::MatrixXd dense_Hff = Hff.toDense();
  // Eigen::MatrixXd dense_Sll = Hllinv.toDense();

  // std::cout << "Sll: " << std::endl << dense_Sll << std::endl;

  // Eigen::MatrixXd error;
  // std::cout << hessian.block(0, 0, lm_dim, lm_dim) << std::endl;
  // std::cout << std::endl;
  // std::cout << dense_Hll << std::endl;
  // error = dense_Hll - hessian.block(0, 0, lm_dim, lm_dim);
  // printf("Hll error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());

  // std::cout << hessian.block(0, lm_dim, lm_dim, rf_dim) << std::endl;
  // std::cout << std::endl;
  // std::cout << dense_Hlf << std::endl;
  // error = dense_Hlf - hessian.block(0, lm_dim, lm_dim, rf_dim);
  // printf("Hlf error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());

  // std::cout << hessian.block(lm_dim, lm_dim, rf_dim, rf_dim) << std::endl;
  // std::cout << std::endl;
  // std::cout << dense_Hff << std::endl;
  // error = dense_Hff - hessian.block(lm_dim, lm_dim, rf_dim, rf_dim);
  // printf("Hff error: Max: %lf, Min: %lf\n", error.maxCoeff(), error.minCoeff());

  // MatrixVisualizer mvis;
  // mvis.render(hessian, "/home/summervibe/Desktop/uav_ws/src/uav_lsm/map_creator/img/RecoverHessian.jpg");
}

void NFRSolver::updateNF(const std::vector<Eigen::MatrixXd>& rSll_blocks, const Eigen::MatrixXd& rSlf, const Eigen::MatrixXd& rSff) {
  assert(rSll_blocks.size() == Sll_blocks_.size());
  std::vector<Eigen::MatrixXd> Ell(rSll_blocks.size());
  Eigen::MatrixXd Elf, Eff;
  for(int index = 0; index < Sll_blocks_.size(); ++index) {
    Ell[index] = Sll_blocks_[index] - rSll_blocks[index];
    // std::cout << "Sll error, index: " << index << std::endl << Ell[index] << std::endl;
  }
  Elf = Slf_ - rSlf;
  Eff = Sff_ - rSff;
  // std::cout << "Elf: " << std::endl << Elf << std::endl;
  // std::cout << "Eff: " << std::endl << Eff << std::endl;

  std::cout << "line factor" << std::endl;
  for(auto &factor: line_factors_) {

    uint64_t ldim = getLineDim(factor.lm_ptr);
    uint64_t lbdim = getLineBlockDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    // Eigen::MatrixXd Jl = factor.sqrt_info * factor.Jl;
    // Eigen::MatrixXd Jf = factor.sqrt_info * factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;

    Eigen::Ref<Eigen::MatrixXd> rEll = Ell[lbdim];
    Eigen::Ref<Eigen::MatrixXd> rElf = Elf.block<LINE_BLOCK_SIZE, POSE_BLOCK_SIZE>(ldim, fdim);
    Eigen::Ref<Eigen::MatrixXd> rEff = Eff.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

    // std::cout << "Jl: " << std::endl << Jl << std::endl;
    // std::cout << "Jf: " << std::endl << Jf << std::endl;
    // std::cout << "Sll: " << std::endl << Sll_blocks_[lbdim] << std::endl;
    // std::cout << "rSll: " << std::endl << rSll_blocks[lbdim] << std::endl;
    // std::cout << "rEll: " << std::endl << rEll << std::endl;
    // std::cout << "rElf: " << std::endl << rElf << std::endl;
    // std::cout << "rEff: " << std::endl << rEff << std::endl;

    Eigen::MatrixXd grad = Jl * rEll * Jl.transpose() + Jl * rElf * Jf.transpose() + Jf * rElf.transpose() * Jl.transpose() + Jf * rEff * Jf.transpose();
    factor.grad = sym2vec(grad);
    Eigen::VectorXd dvec = -factor.grad / factor.grad.norm();

    std::cout << "grad: " << std::endl << grad << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    std::cout << "next info: " << vec2sym(factor.info + dvec) << std::endl;
  }

  std::cout << "surf factor" << std::endl;
  for(auto &factor: surf_factors_) {
    uint64_t sdim = getSurfDim(factor.lm_ptr);
    uint64_t sbdim = getSurfBlockDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    // Eigen::MatrixXd Jl = factor.sqrt_info * factor.Jl;
    // Eigen::MatrixXd Jf = factor.sqrt_info * factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;

    Eigen::Ref<Eigen::MatrixXd> rEll = Ell[sbdim];
    Eigen::Ref<Eigen::MatrixXd> rElf = Elf.block<SURF_BLOCK_SIZE, POSE_BLOCK_SIZE>(sdim, fdim);
    Eigen::Ref<Eigen::MatrixXd> rEff = Eff.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

    // std::cout << "Jl: " << std::endl << Jl << std::endl;
    // std::cout << "Jf: " << std::endl << Jf << std::endl;
    // std::cout << "eSll: " << std::endl << Sll << std::endl;
    // std::cout << "rEll: " << std::endl << rEll << std::endl;
    // std::cout << "rElf: " << std::endl << rElf << std::endl;
    // std::cout << "rEff: " << std::endl << rEff << std::endl;

    Eigen::MatrixXd grad = Jl * rEll * Jl.transpose() + Jl * rElf * Jf.transpose() + Jf * rElf.transpose() * Jl.transpose() + Jf * rEff * Jf.transpose();
    factor.grad = sym2vec(grad);
    Eigen::VectorXd dvec = -factor.grad / factor.grad.norm();

    std::cout << "grad: " << std::endl << grad << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    std::cout << "next info: " << vec2sym(factor.info + dvec) << std::endl;
  }

  for(auto &factor: rel_pose_factors_) {
    uint64_t fidim = getRetainFrameDim(factor.fi_ptr) - getSurfDimOffset();
    uint64_t fjdim = getRetainFrameDim(factor.fj_ptr) - getSurfDimOffset();
    // Eigen::MatrixXd Jl = factor.sqrt_info * factor.Jl;
    // Eigen::MatrixXd Jf = factor.sqrt_info * factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Jfi = factor.Jfi;
    Eigen::Ref<Eigen::MatrixXd> Jfj = factor.Jfj;

    Eigen::Ref<Eigen::MatrixXd> rEii = Eff.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fidim);
    Eigen::Ref<Eigen::MatrixXd> rEij = Eff.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fjdim);
    Eigen::Ref<Eigen::MatrixXd> rEjj = Eff.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fjdim, fjdim);

    // std::cout << "Jl: " << std::endl << Jl << std::endl;
    // std::cout << "Jf: " << std::endl << Jf << std::endl;
    // std::cout << "eSll: " << std::endl << Sll << std::endl;
    // std::cout << "rEll: " << std::endl << rEll << std::endl;
    // std::cout << "rElf: " << std::endl << rElf << std::endl;
    // std::cout << "rEff: " << std::endl << rEff << std::endl;

    Eigen::MatrixXd grad = Jfi * rEii * Jfi.transpose() + Jfi * rEij * Jfj.transpose() + Jfj * rEij.transpose() * Jfi.transpose() + Jfj * rEjj * Jfj.transpose();
    factor.grad = sym2vec(grad);
    Eigen::VectorXd dvec = -factor.grad / factor.grad.norm();

    std::cout << "grad: " << std::endl << grad << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    std::cout << "next info: " << vec2sym(factor.info + dvec) << std::endl;
  }

  for(auto &factor: prior_factors_) {
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();

    // Eigen::MatrixXd Jl = factor.sqrt_info * factor.Jl;
    // Eigen::MatrixXd Jf = factor.sqrt_info * factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> rEii = Eff.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

    // std::cout << "Jl: " << std::endl << Jl << std::endl;
    // std::cout << "Jf: " << std::endl << Jf << std::endl;
    // std::cout << "eSll: " << std::endl << Sll << std::endl;
    // std::cout << "rEll: " << std::endl << rEll << std::endl;
    // std::cout << "rElf: " << std::endl << rElf << std::endl;
    // std::cout << "rEff: " << std::endl << rEff << std::endl;

    Eigen::MatrixXd grad = Jf * rEii * Jf.transpose();
    factor.grad = sym2vec(grad);
    Eigen::VectorXd dvec = -factor.grad / factor.grad.norm();

    std::cout << "grad: " << std::endl << grad << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    std::cout << "next info: " << std::endl << vec2sym(factor.info + dvec) << std::endl;
  }
}

void NFRSolver::solve() {
  // init all factors' info
  // set Identity
  // for(auto &factor: line_factors_) {
  //   factor.sqrt_info = Eigen::MatrixXd::Identity(4, 4) * 0.1;
  // }
  // for(auto &factor: surf_factors_) {
  //   factor.sqrt_info = Eigen::MatrixXd::Identity(3, 3) * 0.1;
  // }
  // for(auto &factor: rel_pose_factors_) {
  //   factor.sqrt_info = Eigen::MatrixXd::Identity(6, 6) * 0.1;
  // }
  
  // for(int i = 0; i < Sll_blocks_.size(); ++i) {
  //   std::cout << "index: " << i << std::endl << Sll_blocks_[i] << std::endl;
  // }
  // std::cout << "Slf: " << std::endl << Slf_ << std::endl;
  // std::cout << "Sff: " << std::endl << Sff_ << std::endl;

  // set closed-form
  printf("[NFRSolver] Set initial guess from closed-form solution!\n");
  std::cout << "line factor initialization" << std::endl;
  for(auto &factor: line_factors_) {
    uint64_t ldim = getLineDim(factor.lm_ptr);
    uint64_t lbdim = getLineBlockDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;

    // std::cout << "Jl: " << std::endl << Jl << std::endl;
    // std::cout << "Jf: " << std::endl << Jf << std::endl;

    Eigen::Ref<Eigen::MatrixXd> Sll = Sll_blocks_[lbdim];
    Eigen::Ref<Eigen::MatrixXd> Slf = Slf_.block<LINE_BLOCK_SIZE, POSE_BLOCK_SIZE>(ldim, fdim);
    Eigen::Ref<Eigen::MatrixXd> Sff = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

    // std::cout << "Sll: " << std::endl << Sll << std::endl;
    // std::cout << "Slf: " << std::endl << Slf << std::endl;
    // std::cout << "Sff: " << std::endl << Sff << std::endl;
    
    Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();
    Eigen::MatrixXd info = sigma.inverse();
    factor.info = sym2vec(info);
    std::cout << "init info: " << std::endl << info << std::endl << std::endl;
    factor.sqrt_info = Eigen::LLT<Eigen::MatrixXd>(info).matrixL().transpose();  
  }
  std::cout << "surf factor initialization" << std::endl;
  for(auto &factor: surf_factors_) {
    uint64_t sdim = getSurfDim(factor.lm_ptr);
    uint64_t sbdim = getSurfBlockDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;

    // std::cout << "Jl: " << std::endl << Jl << std::endl;
    // std::cout << "Jf: " << std::endl << Jf << std::endl;

    Eigen::Ref<Eigen::MatrixXd> Sll = Sll_blocks_[sbdim];
    Eigen::Ref<Eigen::MatrixXd> Slf = Slf_.block<SURF_BLOCK_SIZE, POSE_BLOCK_SIZE>(sdim, fdim);
    Eigen::Ref<Eigen::MatrixXd> Sff = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

    // std::cout << "Sll: " << std::endl << Sll << std::endl;
    // std::cout << "Slf: " << std::endl << Slf << std::endl;
    // std::cout << "Sff: " << std::endl << Sff << std::endl;
    
    Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();
    Eigen::MatrixXd info = sigma.inverse();
    factor.info = sym2vec(info);
    std::cout << "init info: " << std::endl << info << std::endl << std::endl;
    factor.sqrt_info = Eigen::LLT<Eigen::MatrixXd>(info).matrixL().transpose();  
  }
  std::cout << "rel pose factor initialization" << std::endl;
  for(auto &factor: rel_pose_factors_) {
    uint64_t fidim = getRetainFrameDim(factor.fi_ptr) - getSurfDimOffset();
    uint64_t fjdim = getRetainFrameDim(factor.fj_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jfi = factor.Jfi;
    Eigen::Ref<Eigen::MatrixXd> Jfj = factor.Jfj;

    Eigen::Ref<Eigen::MatrixXd> Sii = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fidim);
    Eigen::Ref<Eigen::MatrixXd> Sij = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fjdim);
    Eigen::Ref<Eigen::MatrixXd> Sjj = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fjdim, fjdim);

    // std::cout << "Sii: " << std::endl << Sii << std::endl;
    // std::cout << "Sij: " << std::endl << Sij << std::endl;
    // std::cout << "Sjj: " << std::endl << Sjj << std::endl;

    Eigen::MatrixXd sigma = Jfi * Sii * Jfi.transpose() + Jfi * Sij * Jfj.transpose() + Jfj * Sij.transpose() * Jfi.transpose() + Jfj * Sjj * Jfj.transpose();
    Eigen::MatrixXd info = sigma.inverse();
    factor.info = sym2vec(info);
    std::cout << "init info: " << std::endl << info << std::endl << std::endl;
    factor.sqrt_info = Eigen::LLT<Eigen::MatrixXd>(info).matrixL().transpose();  
  } 
  std::cout << "prior factor initialization" << std::endl;
  for(auto &factor: prior_factors_) {
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Sii = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);
    // std::cout << "Sii: " << std::endl << Sii << std::endl;
    // std::cout << "Sij: " << std::endl << Sij << std::endl;
    // std::cout << "Sjj: " << std::endl << Sjj << std::endl;

    Eigen::MatrixXd sigma = Jf * Sii * Jf.transpose();
    Eigen::MatrixXd info = sigma.inverse();
    factor.info = sym2vec(info);
    std::cout << "init info: " << std::endl << info << std::endl << std::endl;
    factor.sqrt_info = Eigen::LLT<Eigen::MatrixXd>(info).matrixL().transpose();  
  } 


  // std::cout << "margSigma: " << std::endl << marg_covariance_ << std::endl;

  std::vector<Eigen::MatrixXd> rSll_blocks;
  Eigen::MatrixXd rSlf, rSff;
  std::cout << "-------------------------- buildTargetSigma -----------------------------" << std::endl;
  buildTargetSigma(rSll_blocks, rSlf, rSff);
  updateNF(rSll_blocks, rSlf, rSff);
  // solveDense();
}

void NFRSolver::solveCF() {
  int residual_size = line_factors_.size() * LINE_BLOCK_SIZE + surf_factors_.size() * SURF_BLOCK_SIZE + \
    rel_pose_factors_.size() * POSE_BLOCK_SIZE + prior_factors_.size() * POSE_BLOCK_SIZE;
  printf("[NFRSolver] Using closed-form solution! Residual Size: %d, State Parameters' Size: %d\n", residual_size, retain_parameter_size_);
  assert(residual_size == retain_parameter_size_);

  double decay_factor = 1.0;

  for(auto &factor: line_factors_) {
    uint64_t ldim = getLineDim(factor.lm_ptr);
    uint64_t lbdim = getLineBlockDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Sll = Sll_blocks_[lbdim];
    Eigen::Ref<Eigen::MatrixXd> Slf = Slf_.block<LINE_BLOCK_SIZE, POSE_BLOCK_SIZE>(ldim, fdim);
    Eigen::Ref<Eigen::MatrixXd> Sff = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);
    Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();

    // std::cout << Sll_blocks_[lbdim] << std::endl;
    // Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
    // Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse(), 0)).cwiseSqrt();
    // factor.sqrt_info = saes.eigenvectors() * Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
    Eigen::VectorXd Sinvsq = Eigen::VectorXd(saes.eigenvalues().array().inverse()).cwiseSqrt();
    factor.sqrt_info = saes.eigenvectors() * Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

    factor.sqrt_info *= decay_factor;
    // Eigen::MatrixXd info = sigma.inverse();
    // // factor.info = sym2vec(info);
    // // std::cout << "init info: " << std::endl << info << std::endl << std::endl;
    // factor.sqrt_info = Eigen::LLT<Eigen::MatrixXd>(info).matrixL().transpose(); 
  }
  for(auto &factor: surf_factors_) {
    uint64_t sdim = getSurfDim(factor.lm_ptr);
    uint64_t sbdim = getSurfBlockDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Sll = Sll_blocks_[sbdim];
    Eigen::Ref<Eigen::MatrixXd> Slf = Slf_.block<SURF_BLOCK_SIZE, POSE_BLOCK_SIZE>(sdim, fdim);
    Eigen::Ref<Eigen::MatrixXd> Sff = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);
    Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
    Eigen::VectorXd Sinvsq = Eigen::VectorXd(saes.eigenvalues().array().inverse()).cwiseSqrt();
    factor.sqrt_info =  saes.eigenvectors() * Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

    factor.sqrt_info *= decay_factor;

  }
  std::cout << "rel pose factor initialization" << std::endl;
  for(auto &factor: rel_pose_factors_) {
    uint64_t fidim = getRetainFrameDim(factor.fi_ptr) - getSurfDimOffset();
    uint64_t fjdim = getRetainFrameDim(factor.fj_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jfi = factor.Jfi;
    Eigen::Ref<Eigen::MatrixXd> Jfj = factor.Jfj;
    Eigen::Ref<Eigen::MatrixXd> Sii = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fidim);
    Eigen::Ref<Eigen::MatrixXd> Sij = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fjdim);
    Eigen::Ref<Eigen::MatrixXd> Sjj = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fjdim, fjdim);
    Eigen::MatrixXd sigma = Jfi * Sii * Jfi.transpose() + Jfi * Sij * Jfj.transpose() + Jfj * Sij.transpose() * Jfi.transpose() + Jfj * Sjj * Jfj.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
    Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse(), 0)).cwiseSqrt();
    factor.sqrt_info = saes.eigenvectors() * Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

    factor.sqrt_info *= decay_factor;
  } 
  std::cout << "prior factor initialization" << std::endl;
  for(auto &factor: prior_factors_) {
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr) - getSurfDimOffset();
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    Eigen::Ref<Eigen::MatrixXd> Sii = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);
    Eigen::MatrixXd sigma = Jf * Sii * Jf.transpose();

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
    Eigen::VectorXd Sinvsq = Eigen::VectorXd(saes.eigenvalues().array().inverse()).cwiseSqrt();
    factor.sqrt_info = saes.eigenvectors() * Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

    factor.sqrt_info *= decay_factor;
  }
  printf("[NFRSolver] Solving Done!\n");
}

void NFRSolver::solveDense() {
  int residual_size = line_factors_.size() * LINE_BLOCK_SIZE + surf_factors_.size() * SURF_BLOCK_SIZE + \
    rel_pose_factors_.size() * POSE_BLOCK_SIZE + prior_factors_.size() * POSE_BLOCK_SIZE;

  Eigen::MatrixXd J(residual_size, retain_parameter_size_);
  std::cout << "residual_size: " << residual_size << std::endl;
  std::cout << "retain_parameter_size_: " << retain_parameter_size_ << std::endl;
  J.setZero();
  int residual_dim = 0;
  for(auto &factor: line_factors_) {
    uint64_t ldim = getLineDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr);
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    J.block(residual_dim, ldim, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE) = Jl;
    J.block(residual_dim, fdim, LINE_BLOCK_SIZE, POSE_BLOCK_SIZE) = Jf;
    residual_dim += LINE_BLOCK_SIZE;
  }
  for(auto &factor: surf_factors_) {
    uint64_t sdim = getSurfDim(factor.lm_ptr);
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr);
    Eigen::Ref<Eigen::MatrixXd> Jl = factor.Jl;
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    J.block(residual_dim, sdim, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE) = Jl;
    J.block(residual_dim, fdim, SURF_BLOCK_SIZE, POSE_BLOCK_SIZE) = Jf;
    residual_dim += SURF_BLOCK_SIZE;
  }
  for(auto &factor: rel_pose_factors_) {
    uint64_t fidim = getRetainFrameDim(factor.fi_ptr);
    uint64_t fjdim = getRetainFrameDim(factor.fj_ptr);
    Eigen::Ref<Eigen::MatrixXd> Jfi = factor.Jfi;
    Eigen::Ref<Eigen::MatrixXd> Jfj = factor.Jfj;

    J.block(residual_dim, fidim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) = Jfi;
    J.block(residual_dim, fjdim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) = Jfj;
    residual_dim += POSE_BLOCK_SIZE;
  }
  for(auto &factor: prior_factors_) {
    uint64_t fdim = getRetainFrameDim(factor.frame_ptr);
    Eigen::Ref<Eigen::MatrixXd> Jf = factor.Jf;
    std::cout << residual_dim << std::endl;
    std::cout << Jf << std::endl;
    J.block(residual_dim, fdim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) = Jf;
    residual_dim += POSE_BLOCK_SIZE;
  }
  // std::cout << "Full Jacobian: " << std::endl << J << std::endl;
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(J);
  Eigen::MatrixXd Jinv = J.inverse();

  std::cout << "Recover Sigma" << std::endl;
  Eigen::MatrixXd rSigma = J * marg_covariance_ * J.transpose();
  Eigen::MatrixXd rInfo(rSigma.rows(), rSigma.cols());
  rInfo.setZero();
  residual_dim = 0;
  for(int i = 0; i < line_factors_.size(); ++i) {
    rInfo.block(residual_dim, residual_dim, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE) = rSigma.block(residual_dim, residual_dim, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE).inverse();
    residual_dim += LINE_BLOCK_SIZE;
  }
  for(int i = 0; i < surf_factors_.size(); ++i) {
    rInfo.block(residual_dim, residual_dim, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE) = rSigma.block(residual_dim, residual_dim, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE).inverse();
    residual_dim += SURF_BLOCK_SIZE;
  }
  for(int i = 0; i < rel_pose_factors_.size(); ++i) {
    rInfo.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) = rSigma.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE).inverse();
    residual_dim += POSE_BLOCK_SIZE;
  }
  for(int i = 0; i < prior_factors_.size(); ++i) {
    rInfo.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) = rSigma.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE).inverse();
    residual_dim += POSE_BLOCK_SIZE;
  }



  Eigen::MatrixXd rGrad = J * (marg_covariance_ - (J.transpose() * rInfo * J).inverse()) * J.transpose();
  std::cout << rSigma << std::endl;
  residual_dim = 0;
  for(auto &factor: line_factors_) {
    Eigen::MatrixXd dinfo = rSigma.block(residual_dim, residual_dim, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE).inverse();
    std::cout << "dinfo: " << std::endl << dinfo << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    // std::cout << "dgrad: " << std::endl << rGrad.block(residual_dim, residual_dim, LINE_BLOCK_SIZE, LINE_BLOCK_SIZE) << std::endl;
    std::cout << "grad: " << std::endl << factor.grad << std::endl;
    // uint64_t lbdim = getLineBlockDim(factor.lm_ptr);
    // std::cout << "lbdim: " << lbdim << std::endl;
    // std::cout << "Sll: " << lbdim << std::endl;
    residual_dim += LINE_BLOCK_SIZE;
  }
  for(auto &factor: surf_factors_) {
    Eigen::MatrixXd dinfo = rSigma.block(residual_dim, residual_dim, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE).inverse();
    std::cout << "dinfo: " << std::endl << dinfo << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    // std::cout << "dgrad: " << std::endl << rGrad.block(residual_dim, residual_dim, SURF_BLOCK_SIZE, SURF_BLOCK_SIZE) << std::endl;
    std::cout << "grad: " << std::endl << factor.grad << std::endl;
    // uint64_t lbdim = getLineBlockDim(factor.lm_ptr);
    // std::cout << "lbdim: " << lbdim << std::endl;
    // std::cout << "Sll: " << lbdim << std::endl;
    residual_dim += SURF_BLOCK_SIZE;
  }
  for(auto &factor: rel_pose_factors_) {
    Eigen::MatrixXd dinfo = rSigma.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE).inverse();
    std::cout << "dinfo: " << std::endl << dinfo << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    // std::cout << "dgrad: " << std::endl << rGrad.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) << std::endl;
    std::cout << "grad: " << std::endl << factor.grad << std::endl;
    // uint64_t lbdim = getLineBlockDim(factor.lm_ptr);
    // std::cout << "lbdim: " << lbdim << std::endl;
    // std::cout << "Sll: " << lbdim << std::endl;
    residual_dim += POSE_BLOCK_SIZE;
  }

  for(auto &factor: prior_factors_) {
    Eigen::MatrixXd dinfo = rSigma.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE).inverse();
    std::cout << "dinfo: " << std::endl << dinfo << std::endl;
    std::cout << "info: " << std::endl << factor.sqrt_info.transpose() * factor.sqrt_info << std::endl;
    // std::cout << "dgrad: " << std::endl << rGrad.block(residual_dim, residual_dim, POSE_BLOCK_SIZE, POSE_BLOCK_SIZE) << std::endl;
    std::cout << "grad: " << std::endl << factor.grad << std::endl;
    // uint64_t lbdim = getLineBlockDim(factor.lm_ptr);
    // std::cout << "lbdim: " << lbdim << std::endl;
    // std::cout << "Sll: " << lbdim << std::endl;
    residual_dim += POSE_BLOCK_SIZE;
  }
}

LineOB::Ptr NFRSolver::recoverFrameToLine(const Frame::Ptr& frame, const LineLM::Ptr& lm, double decay_factor) {
  uint64_t ldim = getLineDim(lm);
  uint64_t lbdim = getLineBlockDim(lm);
  uint64_t fdim = getRetainFrameDim(frame) - getSurfDimOffset();

  int point_num;
  std::vector<Eigen::Vector3d> virtual_obs;
  Eigen::MatrixXd Jf, Jl;
  evalFrameToLine(frame, lm, Jf, Jl, virtual_obs, point_num);

  Eigen::Ref<Eigen::MatrixXd> Sll = Sll_blocks_[lbdim];
  Eigen::Ref<Eigen::MatrixXd> Slf = Slf_.block<LINE_BLOCK_SIZE, POSE_BLOCK_SIZE>(ldim, fdim);
  Eigen::Ref<Eigen::MatrixXd> Sff = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

  Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
  Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse() * decay_factor, 0)).cwiseSqrt();
  Eigen::MatrixXd sqrt_info = Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

  LineOB::Ptr line_ob = LineOB::Ptr(new LineOB(lm->semantic_type(), virtual_obs[0], virtual_obs[1]));
  line_ob->setSqrtInfo(sqrt_info);
  return line_ob;
}

LineOB::Ptr NFRSolver::recoverFrameToLineDense(const Frame::Ptr& frame, const LineLM::Ptr& lm, double decay_factor) {
  uint64_t ldim = getLineDim(lm);
  uint64_t lbdim = getLineBlockDim(lm);
  uint64_t fdim = getRetainFrameDim(frame);

  int point_num;
  std::vector<Eigen::Vector3d> virtual_obs;
  Eigen::MatrixXd Jf, Jl;
  evalFrameToLine(frame, lm, Jf, Jl, virtual_obs, point_num);

  Eigen::Ref<Eigen::MatrixXd> Sll = covariance_.block<LINE_BLOCK_SIZE, LINE_BLOCK_SIZE>(ldim, ldim);
  Eigen::Ref<Eigen::MatrixXd> Slf = covariance_.block<LINE_BLOCK_SIZE, POSE_BLOCK_SIZE>(ldim, fdim);
  Eigen::Ref<Eigen::MatrixXd> Sff = covariance_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

  Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
  Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse(), 0)).cwiseSqrt();
  Eigen::MatrixXd sqrt_info = Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

  LineOB::Ptr line_ob = LineOB::Ptr(new LineOB(lm->semantic_type(), virtual_obs[0], virtual_obs[1]));
  line_ob->setSqrtInfo(sqrt_info);
  return line_ob;
}


SurfaceOB::Ptr NFRSolver::recoverFrameToSurf(const Frame::Ptr& frame, const SurfaceLM::Ptr& lm, double decay_factor) {
  uint64_t sdim = getSurfDim(lm);
  uint64_t sbdim = getSurfBlockDim(lm);
  uint64_t fdim = getRetainFrameDim(frame) - getSurfDimOffset();

  int point_num;
  std::vector<Eigen::Vector3d> virtual_obs;
  Eigen::MatrixXd Jf, Jl;
  evalFrameToSurf(frame, lm, Jf, Jl, virtual_obs, point_num);

  Eigen::Ref<Eigen::MatrixXd> Sll = Sll_blocks_[sbdim];
  Eigen::Ref<Eigen::MatrixXd> Slf = Slf_.block<SURF_BLOCK_SIZE, POSE_BLOCK_SIZE>(sdim, fdim);
  Eigen::Ref<Eigen::MatrixXd> Sff = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);

  Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
  Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse() * decay_factor, 0)).cwiseSqrt();
  Eigen::MatrixXd sqrt_info = Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

  Eigen::Vector3d centroid = (virtual_obs[0] + virtual_obs[2]) / 2.0;
  Eigen::Vector3d normal = (virtual_obs[0] - centroid).normalized().cross((virtual_obs[1] - centroid).normalized());
  SurfaceOB::Ptr surf_ob = SurfaceOB::Ptr(new SurfaceOB(lm->semantic_type(), centroid, normal, virtual_obs, lm->getRadius(), lm->getRadius()));
  surf_ob->setSqrtInfo(sqrt_info);
  return surf_ob;
}

SurfaceOB::Ptr NFRSolver::recoverFrameToSurfDense(const Frame::Ptr& frame, const SurfaceLM::Ptr& lm, double decay_factor) {
  uint64_t sdim = getSurfDim(lm);
  uint64_t sbdim = getSurfBlockDim(lm);
  uint64_t fdim = getRetainFrameDim(frame);

  int point_num;
  std::vector<Eigen::Vector3d> virtual_obs;
  Eigen::MatrixXd Jf, Jl;
  evalFrameToSurf(frame, lm, Jf, Jl, virtual_obs, point_num);

  Eigen::Ref<Eigen::MatrixXd> Sll = marg_covariance_.block<SURF_BLOCK_SIZE, SURF_BLOCK_SIZE>(sdim, sdim);
  Eigen::Ref<Eigen::MatrixXd> Slf = marg_covariance_.block<SURF_BLOCK_SIZE, POSE_BLOCK_SIZE>(sdim, fdim);
  Eigen::Ref<Eigen::MatrixXd> Sff = marg_covariance_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fdim, fdim);
  Eigen::MatrixXd sigma = Jl * Sll * Jl.transpose() + Jl * Slf * Jf.transpose() + Jf * Slf.transpose() * Jl.transpose() + Jf * Sff * Jf.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
  Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse(), 0)).cwiseSqrt();
  Eigen::MatrixXd sqrt_info = Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

  Eigen::Vector3d centroid = lm->centroid(), normal = lm->normal();
  SurfaceOB::Ptr surf_ob = SurfaceOB::Ptr(new SurfaceOB(lm->semantic_type(), lm->centroid(), lm->normal(), virtual_obs, lm->getRadius(), lm->getRadius()));
  surf_ob->setSqrtInfo(sqrt_info);
  return surf_ob;
}

RelPoseInfo NFRSolver::recoverFrameToFrame(const Frame::Ptr& fi, const Frame::Ptr& fj, double decay_factor) {
  uint64_t fidim = getRetainFrameDim(fi) - getSurfDimOffset();
  uint64_t fjdim = getRetainFrameDim(fj) - getSurfDimOffset();

  Transform virtual_ob;
  Eigen::MatrixXd Jfi, Jfj;
  evalFrameToFrame(fi, fj, Jfi, Jfj, virtual_ob);

  Eigen::Ref<Eigen::MatrixXd> Sii = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fidim);
  Eigen::Ref<Eigen::MatrixXd> Sij = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fjdim);
  Eigen::Ref<Eigen::MatrixXd> Sjj = Sff_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fjdim, fjdim);
  Eigen::MatrixXd sigma = Jfi * Sii * Jfi.transpose() + Jfi * Sij * Jfj.transpose() + Jfj * Sij.transpose() * Jfi.transpose() + Jfj * Sjj * Jfj.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
  Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse() * decay_factor, 0)).cwiseSqrt();
  Eigen::MatrixXd sqrt_info = Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

  return RelPoseInfo(fi, fj, virtual_ob, sqrt_info);
}

RelPoseInfo NFRSolver::recoverFrameToFrameDense(const Frame::Ptr& fi, const Frame::Ptr& fj, double decay_factor) {
  uint64_t fidim = getRetainFrameDim(fi);
  uint64_t fjdim = getRetainFrameDim(fj);

  Transform virtual_ob;
  Eigen::MatrixXd Jfi, Jfj;
  evalFrameToFrame(fi, fj, Jfi, Jfj, virtual_ob);

  Eigen::Ref<Eigen::MatrixXd> Sii = marg_covariance_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fidim);
  Eigen::Ref<Eigen::MatrixXd> Sij = marg_covariance_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fidim, fjdim);
  Eigen::Ref<Eigen::MatrixXd> Sjj = marg_covariance_.block<POSE_BLOCK_SIZE, POSE_BLOCK_SIZE>(fjdim, fjdim);
  Eigen::MatrixXd sigma = Jfi * Sii * Jfi.transpose() + Jfi * Sij * Jfj.transpose() + Jfj * Sij.transpose() * Jfi.transpose() + Jfj * Sjj * Jfj.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(sigma);
  Eigen::VectorXd Sinvsq = Eigen::VectorXd((saes.eigenvalues().array() > 1e-12).select(saes.eigenvalues().array().inverse(), 0)).cwiseSqrt();
  Eigen::MatrixXd sqrt_info = Sinvsq.asDiagonal() * saes.eigenvectors().transpose();

  return RelPoseInfo(fi, fj, virtual_ob, sqrt_info);
}


void NFRSolver::writeBlock(Eigen::SparseMatrix<double>& mat, const Eigen::MatrixXd& block, int row_offset, int col_offset, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      mat.coeffRef(row_offset + i, col_offset + j) = block(i, j);
    }
  }
}


void NFRSolver::writeBlockRowMajor(Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat, const Eigen::MatrixXd& block, int row_offset, int col_offset, int rows, int cols) {
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      mat.coeffRef(row_offset + i, col_offset + j) = block(i, j);
    }
  }
}

}
