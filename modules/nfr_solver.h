#ifndef NFR_SOLVER_H
#define NFR_SOLVER_H

#include <vector>
#include <map>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <ceres/ceres.h>

#include "frame.h"
#include "landmark.h"
#include "utility.h"
#include "factor/local_param.h"
#include "factor/laser_edge_factor.h"
#include "factor/laser_surf_factor.h"
#include "factor/prior_pose_factor.h"
#include "factor/relative_pose_factor.h"
#include "spmm/spmm.h"

namespace SLIM {

typedef uint64_t OrderingType;

struct RelPoseInfo {
  typedef Eigen::Matrix<double, 6, 6> Matrix6d;
  Frame::Ptr fi, fj;
  Transform Tij;
  Matrix6d sqrt_info;
  RelPoseInfo(): fi(nullptr), fj(nullptr), Tij(Transform()), sqrt_info(Matrix6d::Identity() * 1/0.05) {}
  RelPoseInfo(const Frame::Ptr& _fi, const Frame::Ptr& _fj, const Transform& _Tij = Transform(), const Matrix6d& _sqrt_info = Matrix6d::Identity() * 1/0.05) 
  : fi(_fi), fj(_fj), Tij(_Tij), sqrt_info(_sqrt_info) {}
};

#define POSE_BLOCK_SIZE  6
#define LINE_BLOCK_SIZE  4
#define SURF_BLOCK_SIZE  3
#define RP_RES_SIZE      6
#define F2L_RES_SIZE     4
#define F2S_RES_SIZE     4

struct FramePriorNF {
  Frame::Ptr frame_ptr;
  Eigen::MatrixXd Jf;
  Transform ob;
  Eigen::MatrixXd sqrt_info;
  Eigen::VectorXd info;
  Eigen::VectorXd grad;
};

struct FrameToLineNF {
  Frame::Ptr frame_ptr;
  LineLM::Ptr lm_ptr;
  Eigen::MatrixXd Jf, Jl;
  std::vector<Eigen::Vector3d> ob;
  Eigen::MatrixXd sqrt_info;
  Eigen::VectorXd info;
  Eigen::VectorXd grad;
  int point_num;
};

struct FrameToSurfNF {
  Frame::Ptr frame_ptr;
  SurfaceLM::Ptr lm_ptr;
  Eigen::MatrixXd Jf, Jl;
  std::vector<Eigen::Vector3d> ob;
  Eigen::MatrixXd sqrt_info;
  Eigen::VectorXd info;
  Eigen::VectorXd grad;
  int point_num;
};

struct FrameToFrameNF {
  Frame::Ptr fi_ptr;
  Frame::Ptr fj_ptr;
  Eigen::MatrixXd Jfi, Jfj;
  Transform ob;
  Eigen::MatrixXd sqrt_info;
  Eigen::VectorXd info;
  Eigen::VectorXd grad;
};

class BaseMat {
 public:
  BaseMat() {}
  ~BaseMat() {}

  Eigen::MatrixXd Ql;
  CuSMat dsP, dsPtVl, dsDVl;
  CuDMat ddPQl, ddQVf, ddPQVf;
};

class NFRSolver {
 public:
  typedef Eigen::SparseMatrix<double> SpMatrix;
  typedef Eigen::TriangularView<SpMatrix, Eigen::Upper> TriangularView;

  NFRSolver() {
    omp_init_lock(&lock);
  };

  ~NFRSolver() {

  }

  void init();

  void evalFramePrior(const Frame::Ptr& frame, Eigen::MatrixXd& Jf, Transform& virtual_ob);

  void evalFrameToFrame(const Frame::Ptr& fi, const Frame::Ptr& fj, Eigen::MatrixXd& Ji, Eigen::MatrixXd& Jj, Transform& virtual_ob);

  void evalFrameToLine(const Frame::Ptr& frame, const LineLM::Ptr& line,
                       Eigen::MatrixXd& Jf, Eigen::MatrixXd& Jl, std::vector<Eigen::Vector3d>& virtual_ob, int& point_num);

  void evalFrameToSurf(const Frame::Ptr& frame, const SurfaceLM::Ptr& surface,
                       Eigen::MatrixXd& Jf, Eigen::MatrixXd& Jl, std::vector<Eigen::Vector3d>& virtual_ob, int& point_num);

  void evalFactor(ceres::CostFunction* cost_func, ceres::LossFunction* loss_func, 
        const std::vector<double*>& parameter_blocks, Eigen::VectorXd& residuals, 
        std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& jacobians);

  void buildDenseStruct();
  
  void buildSparseStruct();

  void calcBaseMat(Eigen::MatrixXd& Q, Eigen::MatrixXd& PQ, Eigen::MatrixXd& PQl, Eigen::MatrixXd& Wl, Eigen::MatrixXd& Wf);

  void calcBaseMatV2(Eigen::MatrixXd& Q, Eigen::MatrixXd& PQ, Eigen::MatrixXd& PQl, Eigen::MatrixXd& Wl, Eigen::MatrixXd& Wf);

  void buildSparseStructCUDA();

  void addFrameToLineNF(const Frame::Ptr& f, const LineLM::Ptr& lm);

  void addFrameToSurfNF(const Frame::Ptr& f, const SurfaceLM::Ptr& lm);

  void addFrameToFrameNF(const Frame::Ptr& fi, const Frame::Ptr& fj);

  void addFramePriorNF(const Frame::Ptr& frame);

  void buildTargetSigma(std::vector<Eigen::MatrixXd>& rSll_blocks, Eigen::MatrixXd& rSlf, Eigen::MatrixXd& rSff);

  void updateNF(const std::vector<Eigen::MatrixXd>& rSll_blocks, const Eigen::MatrixXd& rSlf, const Eigen::MatrixXd& rSff);

  Eigen::VectorXd concatState();

  Eigen::MatrixXd concatGrad();

  void solve();

  void solveCF();
  
  void solveDense();

  void checkAccuracy();

  LineOB::Ptr recoverFrameToLine(const Frame::Ptr& frame, const LineLM::Ptr& lm, double decay_factor = 1.0);

  SurfaceOB::Ptr recoverFrameToSurf(const Frame::Ptr& frame, const SurfaceLM::Ptr& lm, double decay_factor = 1.0);

  RelPoseInfo recoverFrameToFrame(const Frame::Ptr& fi, const Frame::Ptr& fj, double decay_factor = 1.0);

  LineOB::Ptr recoverFrameToLineDense(const Frame::Ptr& frame, const LineLM::Ptr& lm, double decay_factor = 1.0);

  SurfaceOB::Ptr recoverFrameToSurfDense(const Frame::Ptr& frame, const SurfaceLM::Ptr& lm, double decay_factor = 1.0);

  RelPoseInfo recoverFrameToFrameDense(const Frame::Ptr& fi, const Frame::Ptr& fj, double decay_factor = 1.0);

  void writeBlock(Eigen::SparseMatrix<double>& mat, const Eigen::MatrixXd& block, int row_offset, int col_offset, int rows, int cols);

  void writeBlockRowMajor(Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat, const Eigen::MatrixXd& block, int row_offset, int col_offset, int rows, int cols);

  uint64_t getLineDimOffset() {
    return retain_line_block_size_ * LINE_BLOCK_SIZE;
  }

  uint64_t getSurfDimOffset() {
    return getLineDimOffset() + retain_surf_block_size_ * SURF_BLOCK_SIZE;
  }
  
  uint64_t getRetainFrameDimOffset(int offset = 0) {
    return getSurfDimOffset() + retain_frame_block_size_ * (POSE_BLOCK_SIZE + offset);
  }

  uint64_t getMargFrameBlockDim(const Frame::Ptr frame) {
    return retain_line_block_size_ + retain_surf_block_size_ + \
            retain_frame_block_size_ + map_marg_frame_[frame];
  }

  uint64_t getRetainFrameBlockDim(const Frame::Ptr frame) {
    return retain_line_block_size_ + retain_surf_block_size_ + \
            map_retain_frame_[frame];
  }

  uint64_t getLineBlockDim(const LineLM::Ptr lm) {
    return map_retain_line_[lm];
  }

  uint64_t getSurfBlockDim(const SurfaceLM::Ptr lm) {
    return retain_line_block_size_ + map_retain_surf_[lm];
  }

  uint64_t getMargFrameDim(const Frame::Ptr frame, int offset = 0) {
    return getRetainFrameDimOffset(offset) + map_marg_frame_[frame] * (POSE_BLOCK_SIZE + offset);
  }

  uint64_t getRetainFrameDim(const Frame::Ptr frame, int offset = 0) {
    return getSurfDimOffset() + map_retain_frame_[frame] * (POSE_BLOCK_SIZE + offset);
  }

  uint64_t getLineDim(const LineLM::Ptr lm) {
    return map_retain_line_[lm] * LINE_BLOCK_SIZE;
  }

  uint64_t getSurfDim(const SurfaceLM::Ptr lm) {
    return getLineDimOffset() + map_retain_surf_[lm] * SURF_BLOCK_SIZE;
  }

  void addRetainFrame(const Frame::Ptr& frame) {
    map_retain_frame_.insert(std::make_pair(frame, retain_frame_order_));
    retain_frame_order_++;
  }

  void addRetainLine(const LineLM::Ptr& line) {
    map_retain_line_.insert(std::make_pair(line, retain_line_order_));
    retain_line_order_++;
  }

  void addRetainSurf(const SurfaceLM::Ptr& surf) {
    map_retain_surf_.insert(std::make_pair(surf, retain_surf_order_));
    retain_surf_order_++;
  }

  void addMargFrame(const Frame::Ptr& frame) {
    map_marg_frame_.insert(std::make_pair(frame, marg_frame_order_));
    marg_frame_order_++;
  }

  void setPivotFrame(const Frame::Ptr& frame, const Eigen::Matrix<double, 6, 6>& sqrt_info) {
    fixed_frames_.insert(frame);
    pivot_frame_ = frame;
    pivot_sqrt_info_ = sqrt_info;
  }

  void addRelPoseInfo(const RelPoseInfo& info) {
    rel_pose_info_.push_back(info);
  }

  bool isMarginalized(const Frame::Ptr& frame) {
    return map_marg_frame_.find(frame) != map_marg_frame_.end();
  }

  uint64_t getRetainFrameSize() {
    return map_retain_frame_.size();
  }

  uint64_t getRetainLineSize() {
    return map_retain_line_.size();
  }

//  private:

  omp_lock_t lock;

  std::map<Frame::Ptr, OrderingType> map_retain_frame_;
  std::map<LineLM::Ptr, OrderingType> map_retain_line_;
  std::map<SurfaceLM::Ptr, OrderingType> map_retain_surf_;

  std::map<Frame::Ptr, OrderingType> map_marg_frame_;

  Eigen::MatrixXd hessian_, prior_hessian_, prior_cov_, marg_hessian_, inverse_marg_hessian_;
  Eigen::MatrixXd nfr_hessian_;

  Eigen::SparseMatrix<double> A_, Ainv_, U_, C_, Vl_, Vf_, Vm_;
  Eigen::SparseMatrix<double, Eigen::RowMajor, int> Ar_, Ainvr_, Ur_, Cr_, Vlr_, Vfr_, Vmr_;
  Eigen::SparseMatrix<double> Sll_;
  std::vector<Eigen::MatrixXd> Sll_blocks_, marg_Sll_blocks_;
  std::vector<Eigen::MatrixXd> A_blocks_;
  Eigen::MatrixXd Sff_, Smm_, Slf_, Slm_, Sfm_, marg_Slf_, marg_Sff_;
  Eigen::MatrixXd covariance_, marg_covariance_;

  std::set<Frame::Ptr> fixed_frames_;

  Frame::Ptr pivot_frame_;
  Eigen::Matrix<double, 6, 6> pivot_sqrt_info_;
  std::vector<RelPoseInfo> rel_pose_info_;
  // std::vector<NonlinearFactor> factors_;
  std::map<uint64_t, std::map<uint64_t, Eigen::MatrixXd>> jacobians_;

  std::vector<std::pair<const double*, const double*>> line_param_pairs_;
  std::vector<std::pair<const double*, const double*>> surf_param_pairs_;
  std::vector<std::pair<const double*, const double*>> retain_frame_param_pairs_;
  std::vector<std::pair<const double*, const double*>> marg_frame_param_pairs_;
  std::vector<std::pair<const double*, const double*>> line_retain_frame_param_pairs_;
  std::vector<std::pair<const double*, const double*>> surf_retain_frame_param_pairs_;
  std::vector<std::pair<const double*, const double*>> line_marg_frame_param_pairs_;
  std::vector<std::pair<const double*, const double*>> surf_marg_frame_param_pairs_;
  std::vector<std::pair<const double*, const double*>> retain_marg_frame_param_pairs_;

  std::map<const double*, LineLM::Ptr> map_line_ptr_;
  std::map<const double*, SurfaceLM::Ptr> map_surf_ptr_;
  std::map<const double*, Frame::Ptr> map_retain_frame_ptr_;
  std::map<const double*, Frame::Ptr> map_marg_frame_ptr_;

  std::vector<FrameToFrameNF> rel_pose_factors_;
  std::vector<FrameToLineNF> line_factors_;
  std::vector<FrameToSurfNF> surf_factors_;
  std::vector<FramePriorNF> prior_factors_;

  uint64_t retain_frame_order_ = 0;
  uint64_t retain_line_order_ = 0;
  uint64_t retain_surf_order_ = 0;
  uint64_t marg_frame_order_ = 0;

  uint64_t marg_block_size_ = 0;
  uint64_t marg_parameter_size_ = 0;
  uint64_t retain_block_size_ = 0;
  uint64_t retain_parameter_size_ = 0;

  uint64_t retain_frame_block_size_ = 0;
  uint64_t retain_line_block_size_ = 0;
  uint64_t retain_surf_block_size_ = 0;

  std::vector<Eigen::MatrixXd> factor_jacobians_;
};

}


#endif