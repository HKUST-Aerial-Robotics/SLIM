#pragma once

#include <vector>
#include <algorithm>
#include <numeric>

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "device_buffer.h"
#include <cublas_v2.h>
#include <cusolverDn.h>

void printGpuMemInfo();

struct CusparseHandle
{
	CusparseHandle() { init(); }
	~CusparseHandle() { destroy(); }
	void init() { cusparseCreate(&handle); }
	void destroy() { cusparseDestroy(handle); }
	operator cusparseHandle_t() const { return handle; }
	CusparseHandle(const CusparseHandle&) = delete;
	CusparseHandle& operator=(const CusparseHandle&) = delete;
	cusparseHandle_t handle;
};

struct CusparseMatDescriptor
{
	CusparseMatDescriptor() { init(); }
	~CusparseMatDescriptor() { destroy(); }

	void init() {
		cusparseCreateMatDescr(&desc);
		cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
		cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
		cusparseSetMatDiagType(desc, CUSPARSE_DIAG_TYPE_NON_UNIT);
	}

	void destroy() { cusparseDestroyMatDescr(desc); }
	operator cusparseMatDescr_t() const { return desc; }
	CusparseMatDescriptor(const CusparseMatDescriptor&) = delete;
	CusparseMatDescriptor& operator=(const CusparseMatDescriptor&) = delete;
	cusparseMatDescr_t desc;
};

void convertEigenToCuSparse(
    const Eigen::SparseMatrix<double, Eigen::RowMajor, int> &mat, int *row, int *col, double *val);

void convertCuSparseToEigen(
    const int *row,
    const int *col,
    const double *val,
    const int num_non0,
    const int mat_row,
    const int mat_col,
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> &mat);


Eigen::MatrixXd solveCholeskyCUDA(const Eigen::MatrixXd& mat);

std::vector<Eigen::MatrixXd> inverseParallelCUDA(const std::vector<Eigen::MatrixXd>& mats, int m);

class CuDMat {
 public:
  CuDMat(): rows_(0), cols_(0) {}
  ~CuDMat() {
    values_.destroy();
  }

  CuDMat(const int rows, const int cols)
  : rows_(rows), cols_(cols) {
		values_.allocate(rows_ * cols_);
    values_.fillZero();
	}

  CuDMat(const Eigen::MatrixXd& mat) {
		rows_ = mat.rows();
		cols_ = mat.cols();
		values_.allocate(rows_ * cols_);
		values_.upload(mat.data());
	}

  CuDMat(CuDMat&& other) noexcept {
    rows_ = other.rows_;
    cols_ = other.cols_;
    values_.data = other.values_.data;
    values_.size = other.values_.size;
    other.values_.data = nullptr;
    other.values_.size = 0;
  }


  void resize(const int rows, const int cols) {
    rows_ = rows;
    cols_ = cols;
    values_.allocate(rows_ * cols_);
    values_.fillZero();
  }
  
  void release();

  CuDMat& operator=(CuDMat&& other) noexcept;

  CuDMat operator*(const CuDMat& other) const;

  CuDMat operator+(const CuDMat& other) const;
  
  CuDMat operator-(const CuDMat& other) const;

  CuDMat transpose() const;

  void addSelf(const CuDMat& other);

  void minusSelf(const CuDMat& other);

  CuDMat inverse();

  Eigen::MatrixXd toEigenDense() const;

 public:
	int rows_, cols_; 
	DeviceBuffer<double> values_;
};

class CuSMat {
 public:
  CuSMat(): rows_(0), cols_(0), nnz_(0) {}
  ~CuSMat() {
    values_.destroy();
    rowPtr_.destroy();
    colInd_.destroy();
  }

  CuSMat(const int rows, const int cols) 
  : rows_(rows), cols_(cols), nnz_(0) {
    rowPtr_.allocate(rows_ + 1);
  }

  CuSMat(CuSMat&& other) noexcept {
    rows_ = other.rows_;
    cols_ = other.cols_;
    nnz_ = other.nnz_;

    rowPtr_.data = other.rowPtr_.data;
    rowPtr_.size = other.rowPtr_.size;
    other.rowPtr_.data = nullptr;
    other.rowPtr_.size = 0;

    colInd_.data = other.colInd_.data;
    colInd_.size = other.colInd_.size;
    other.colInd_.data = nullptr;
    other.colInd_.size = 0;

    values_.data = other.values_.data;
    values_.size = other.values_.size;
    other.values_.data = nullptr;
    other.values_.size = 0;
  }

  void release();

  CuSMat(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat);

  CuSMat& operator=(CuSMat&& other) noexcept;

  CuSMat operator+(const CuSMat& other) const;

  CuSMat operator-(const CuSMat& other) const;

  CuSMat operator*(const CuSMat& other) const;

  CuDMat operator*(const CuDMat& other) const;

  Eigen::SparseMatrix<double, Eigen::RowMajor, int> toEigenSparse() const;

 public:
	int rows_, cols_, nnz_; 
	DeviceBuffer<double> values_;
	DeviceBuffer<int> rowPtr_;
	DeviceBuffer<int> colInd_;
  // cusparseSpMatDescr_t mat_;
};

void matmul(const CuSMat& A, const CuDMat& B, CuDMat& C);

void matmul(const CuDMat& A, const CuDMat& B, CuDMat& C);

CuDMat matmul3DM(const CuDMat& A, const CuDMat& B, const CuDMat& C);

void matmulAtB(const CuDMat& A, const CuDMat& B, CuDMat& AtB);

void matmulAtXA(const CuDMat& X, const CuDMat& A, CuDMat& AtXA);
