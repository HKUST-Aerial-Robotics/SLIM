#include "spmm.h"
#include <iostream>

void printGpuMemInfo() {
  size_t freeMem, totalMem;
  CHECK_CUDA(cudaMemGetInfo(&freeMem, &totalMem))
  size_t usedMem = totalMem - freeMem;
  printf("Current thread memory usage: %zu bytes\n", usedMem);    
}

void convertEigenToCuSparse(
    const Eigen::SparseMatrix<double, Eigen::RowMajor, int> &mat, int *row, int *col, double *val)
{
  const int num_non0  = mat.nonZeros();
  const int num_outer = mat.rows() + 1;
  cudaMemcpy(row,
             mat.outerIndexPtr(),
             sizeof(int) * num_outer,
             cudaMemcpyHostToDevice);

  cudaMemcpy(
      col, mat.innerIndexPtr(), sizeof(int) * num_non0, cudaMemcpyHostToDevice);

  cudaMemcpy(
      val, mat.valuePtr(), sizeof(double) * num_non0, cudaMemcpyHostToDevice);
}

void convertCuSparseToEigen(
    const int *row,
    const int *col,
    const double *val,
    const int num_non0,
    const int mat_row,
    const int mat_col,
    Eigen::SparseMatrix<double, Eigen::RowMajor, int> &mat)
{
  std::vector<int> outer(mat_row + 1);
  std::vector<int> inner(num_non0);
  std::vector<double> value(num_non0);

  cudaMemcpy(
      outer.data(), row, sizeof(int) * (mat_row + 1), cudaMemcpyDeviceToHost);

  cudaMemcpy(inner.data(), col, sizeof(int) * num_non0, cudaMemcpyDeviceToHost);

  cudaMemcpy(
      value.data(), val, sizeof(double) * num_non0, cudaMemcpyDeviceToHost);

  Eigen::Map<Eigen::SparseMatrix<double, Eigen::RowMajor, int>> mat_map(
      mat_row, mat_col, num_non0, outer.data(), inner.data(), value.data());

  mat = mat_map.eval();
}

CuDMat CuDMat::inverse() {
  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;
  CuDMat result(Eigen::MatrixXd::Identity(rows_, cols_));
  assert(rows_ == cols_);

  int info = 0;
  int lwork = 0;  /* size of workspace */
  double *d_work = nullptr; /* device workspace for getrf */
  int *d_info = nullptr; /* error info */

  CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

  CHECK_CUSOLVER(cusolverDnDgetrf_bufferSize(cusolverH, rows_, cols_, values_.data, rows_, &lwork));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));
  CHECK_CUSOLVER(cusolverDnDgetrf(cusolverH, rows_, cols_, values_.data, rows_, d_work, NULL, d_info));

  // CHECK_CUDA(cudaMemcpyAsync(lu.data, values_.data, sizeof(double) * (rows_ * cols_), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_CUSOLVER(cusolverDnDgetrs(cusolverH, CUBLAS_OP_N, rows_, cols_, /* nrhs */
                                        values_.data, rows_, NULL, result.values_.data, rows_, d_info));

  CHECK_CUDA(cudaFree(d_info));
  CHECK_CUDA(cudaFree(d_work));

  CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return result;
}

void CuDMat::release() {
  rows_ = 0;
  cols_ = 0;
  values_.destroy();
}

CuDMat& CuDMat::operator=(CuDMat&& other) noexcept {
  if (this != &other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    values_.data = other.values_.data;
    values_.size = other.values_.size;
    other.values_.data = nullptr;
    other.values_.size = 0;
  }
  return *this;
}

CuDMat CuDMat::operator*(const CuDMat& other) const {
  cublasHandle_t handle;
  CuDMat result(rows_, other.cols_);
  assert(cols_ == other.rows_);
  int m = rows_, n = other.cols_, k = cols_;
  double alpha = 1.0;
  double beta = 0.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, values_.data, m, other.values_.data, k, &beta, result.values_.data, m))
  CHECK_CUBLAS(cublasDestroy(handle))
  return result;
}

CuDMat CuDMat::operator+(const CuDMat& other) const {
  cublasHandle_t handle;
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  CuDMat result(rows_, other.cols_);
  int m = rows_, n = other.cols_, k = cols_;
  double alpha = 1.0;
  double beta = 1.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, values_.data, rows_, &beta, other.values_.data, rows_, result.values_.data, rows_))
  CHECK_CUBLAS(cublasDestroy(handle))
  return result;
}

CuDMat CuDMat::operator-(const CuDMat& other) const {
  cublasHandle_t handle;
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  CuDMat result(rows_, other.cols_);
  int m = rows_, n = other.cols_, k = cols_;
  double alpha = 1.0;
  double beta = -1.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, values_.data, rows_, &beta, other.values_.data, rows_, result.values_.data, rows_))
  CHECK_CUBLAS(cublasDestroy(handle))
  return result;
}

CuDMat CuDMat::transpose() const {
  cublasHandle_t handle;
  CuDMat result(cols_, rows_);
  CuDMat dummy(cols_, rows_);
  double alpha = 1.0;
  double beta = 0.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, cols_, rows_, &alpha, values_.data, rows_, &beta, dummy.values_.data, cols_, result.values_.data, cols_))
  CHECK_CUBLAS(cublasDestroy(handle))
  return result;
}

void CuDMat::addSelf(const CuDMat& other) {
  cublasHandle_t handle;
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  int m = rows_, n = other.cols_, k = cols_;
  double alpha = 1.0;
  double beta = 1.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, values_.data, rows_, &beta, other.values_.data, rows_, values_.data, rows_))
  CHECK_CUBLAS(cublasDestroy(handle))
}

void CuDMat::minusSelf(const CuDMat& other) {
  cublasHandle_t handle;
  assert(rows_ == other.rows_);
  assert(cols_ == other.cols_);
  int m = rows_, n = other.cols_, k = cols_;
  double alpha = 1.0;
  double beta = -1.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows_, cols_, &alpha, values_.data, rows_, &beta, other.values_.data, rows_, values_.data, rows_))
  CHECK_CUBLAS(cublasDestroy(handle))
}

Eigen::MatrixXd solveCholeskyCUDA(const Eigen::MatrixXd& mat) {
  assert(mat.rows() == mat.cols());
  int m = mat.rows();
  const int batchSize = 1;
  std::vector<int> infoArray(batchSize, 0); /* host copy of error info */
  Eigen::MatrixXd L0(m, m);
  std::vector<double *> Aarray(batchSize, nullptr);
  std::vector<double *> Barray(batchSize, nullptr);
  double **d_Aarray = nullptr;
  double **d_Barray = nullptr;
  int *d_infoArray = nullptr;

  cusolverDnHandle_t cusolverH = NULL;
  cudaStream_t stream = NULL;

  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  /* step 1: create cusolver handle, bind a stream */
  CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&Aarray[0]), sizeof(double) * m * m));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_infoArray), sizeof(int) * infoArray.size()));
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_Aarray), sizeof(double *) * Aarray.size()));
  CHECK_CUDA(cudaMemcpyAsync(Aarray[0], mat.data(), sizeof(double) * mat.size(), cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(double) * Aarray.size(), cudaMemcpyHostToDevice, stream));

  /* step 3: Cholesky factorization */
  CHECK_CUSOLVER(cusolverDnDpotrfBatched(cusolverH, uplo, m, d_Aarray, m, d_infoArray, batchSize));

  CHECK_CUDA(cudaMemcpyAsync(infoArray.data(), d_infoArray, sizeof(int) * infoArray.size(), cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaMemcpyAsync(L0.data(), Aarray[0], sizeof(double) * m * m, cudaMemcpyDeviceToHost, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  /* free resources */
  CHECK_CUDA(cudaFree(d_Aarray));
  CHECK_CUDA(cudaFree(d_Barray));
  CHECK_CUDA(cudaFree(d_infoArray));
  CHECK_CUDA(cudaFree(Aarray[0]));
  CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
  CHECK_CUDA(cudaStreamDestroy(stream));
  return L0.triangularView<Eigen::Lower>();
}

std::vector<Eigen::MatrixXd> inverseParallelCUDA(const std::vector<Eigen::MatrixXd>& mats, int m) {
  
  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate_v2(&handle));
  const int batchSize = mats.size();
  std::vector<double *> Aarray(batchSize, nullptr);
  std::vector<double *> Carray(batchSize, nullptr);
  for(int i = 0; i < batchSize; ++i) {
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&Aarray[i]), sizeof(double) * m * m))
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&Carray[i]), sizeof(double) * m * m))
    // CHECK_CUDA(cudaMemcpyAsync(Aarray[i], mats[i].data(), sizeof(double) * m * m, cudaMemcpyHostToDevice, stream))
    CHECK_CUDA(cudaMemcpy(Aarray[i], mats[i].data(), sizeof(double) * m * m, cudaMemcpyHostToDevice))
  }
  // CHECK_CUDA(cudaStreamSynchronize(stream))
  CHECK_CUDA(cudaDeviceSynchronize())

  int *d_infoArray = nullptr;
  double **d_Aarray = nullptr, **d_Carray;

  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_infoArray), sizeof(int) * batchSize))
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_Aarray), sizeof(double*) * batchSize))
  CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&d_Carray), sizeof(double*) * batchSize))
  CHECK_CUDA(cudaMemcpyAsync(d_Aarray, Aarray.data(), sizeof(double*) * Aarray.size(), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaMemcpyAsync(d_Carray, Carray.data(), sizeof(double*) * Aarray.size(), cudaMemcpyHostToDevice, stream))
  CHECK_CUDA(cudaStreamSynchronize(stream))

  CHECK_CUBLAS(cublasDgetrfBatched(handle, m, d_Aarray, m, NULL, d_infoArray, batchSize))
  CHECK_CUBLAS(cublasDgetriBatched(handle, m, d_Aarray, m, NULL, d_Carray, m, d_infoArray, batchSize))

  std::vector<Eigen::MatrixXd> result(batchSize);
  for(int i = 0; i < batchSize; ++i) {
    result[i] = Eigen::MatrixXd::Identity(m, m);
    // CHECK_CUDA(cudaMemcpyAsync(result[i].data(), Carray[i], sizeof(double) * m * m, cudaMemcpyDeviceToHost, stream))
    CHECK_CUDA(cudaMemcpy(result[i].data(), Carray[i], sizeof(double) * m * m, cudaMemcpyDeviceToHost))
  }
  // CHECK_CUDA(cudaStreamSynchronize(stream));
  CHECK_CUDA(cudaDeviceSynchronize())

  // printGpuMemInfo();

  CHECK_CUDA(cudaFree(d_infoArray));
  for(int i = 0; i < batchSize; ++i) {
    CHECK_CUDA(cudaFree(Aarray[i]));
    CHECK_CUDA(cudaFree(Carray[i]));
  }
  CHECK_CUDA(cudaFree(d_Aarray));
  CHECK_CUDA(cudaFree(d_Carray));
  CHECK_CUBLAS(cublasDestroy(handle))

  return result;
}


CuSMat::CuSMat(const Eigen::SparseMatrix<double, Eigen::RowMajor, int>& mat) {
  rows_ = mat.rows();
  cols_ = mat.cols(); 
  nnz_ = mat.nonZeros();
  rowPtr_.allocate(mat.rows() + 1); 
  values_.allocate(nnz_);
  colInd_.allocate(nnz_);

  convertEigenToCuSparse(mat, rowPtr_.data, colInd_.data, values_.data);

  // cusparseCreateCsr(&mat_, rows_, cols_, nnz_,
  //           rowPtr_.data, colInd_.data, values_.data,
  //           CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
  //           CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
}

void CuSMat::release() {
  // CHECK_CUSPARSE(cusparseDestroySpMat(mat_));
  rowPtr_.destroy();
  colInd_.destroy();
  values_.destroy();
}

CuSMat CuSMat::operator+(const CuSMat& other) const {
  CuSMat result(rows_, other.cols_);

  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle))

  cusparseMatDescr_t matA, matB, matC;   
  double              alpha       = 1.0;
  double              beta        = 1.0;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matA))
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matB))
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matC))

  int baseC, nnzC;
  /* alpha, nnzTotalDevHostPtr points to host memory */
  size_t bufferSizeInBytes;
  char *buffer = NULL;
  int *nnzTotalDevHostPtr = &nnzC;
  CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST))

  /* prepare buffer */
  CHECK_CUSPARSE(cusparseDcsrgeam2_bufferSizeExt(handle, rows_, cols_,
    &alpha,
    matA, nnz_,
    values_.data, rowPtr_.data, colInd_.data,
    &beta,
    matB, other.nnz_,
    other.values_.data, other.rowPtr_.data, other.colInd_.data,
    matC,
    result.values_.data, result.rowPtr_.data, result.colInd_.data,
    &bufferSizeInBytes
  ))

  CHECK_CUDA(cudaMalloc((void**)&buffer, sizeof(char) * bufferSizeInBytes))
  CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, rows_, cols_,
    matA, nnz_, rowPtr_.data, colInd_.data,
    matB, other.nnz_, other.rowPtr_.data, other.colInd_.data,
    matC, result.rowPtr_.data, nnzTotalDevHostPtr,
    buffer))

  if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;
  } 
  else {
    CHECK_CUDA(cudaMemcpy(&nnzC, result.rowPtr_.data + rows_, sizeof(int), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(&baseC, result.rowPtr_.data, sizeof(int), cudaMemcpyDeviceToHost))
    nnzC -= baseC;
  }

  result.values_.allocate(nnzC);
  result.colInd_.allocate(nnzC);
  result.nnz_ = nnzC;

  CHECK_CUSPARSE(cusparseDcsrgeam2(handle, rows_, cols_,
    &alpha,
    matA, nnz_,
    values_.data, rowPtr_.data, colInd_.data,
    &beta,
    matB, other.nnz_,
    other.values_.data, other.rowPtr_.data, other.colInd_.data,
    matC,
    result.values_.data, result.rowPtr_.data, result.colInd_.data,
    buffer))

  CHECK_CUSPARSE(cusparseDestroy(handle))
  return result;
}

CuSMat CuSMat::operator-(const CuSMat& other) const {
  CuSMat result(rows_, other.cols_);

  cusparseHandle_t handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle))

  cusparseMatDescr_t matA, matB, matC;   
  double              alpha       = 1.0;
  double              beta        = -1.0;
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matA))
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matB))
  CHECK_CUSPARSE(cusparseCreateMatDescr(&matC))

  int baseC, nnzC;
  /* alpha, nnzTotalDevHostPtr points to host memory */
  size_t bufferSizeInBytes;
  char *buffer = NULL;
  int *nnzTotalDevHostPtr = &nnzC;
  CHECK_CUSPARSE(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST))

  /* prepare buffer */
  CHECK_CUSPARSE(cusparseDcsrgeam2_bufferSizeExt(handle, rows_, cols_,
    &alpha,
    matA, nnz_,
    values_.data, rowPtr_.data, colInd_.data,
    &beta,
    matB, other.nnz_,
    other.values_.data, other.rowPtr_.data, other.colInd_.data,
    matC,
    result.values_.data, result.rowPtr_.data, result.colInd_.data,
    &bufferSizeInBytes
  ))

  CHECK_CUDA(cudaMalloc((void**)&buffer, sizeof(char) * bufferSizeInBytes))
  CHECK_CUSPARSE(cusparseXcsrgeam2Nnz(handle, rows_, cols_,
    matA, nnz_, rowPtr_.data, colInd_.data,
    matB, other.nnz_, other.rowPtr_.data, other.colInd_.data,
    matC, result.rowPtr_.data, nnzTotalDevHostPtr,
    buffer))

  if (NULL != nnzTotalDevHostPtr){
    nnzC = *nnzTotalDevHostPtr;
  } 
  else {
    CHECK_CUDA(cudaMemcpy(&nnzC, result.rowPtr_.data + rows_, sizeof(int), cudaMemcpyDeviceToHost))
    CHECK_CUDA(cudaMemcpy(&baseC, result.rowPtr_.data, sizeof(int), cudaMemcpyDeviceToHost))
    nnzC -= baseC;
  }

  result.values_.allocate(nnzC);
  result.colInd_.allocate(nnzC);
  result.nnz_ = nnzC;

  CHECK_CUSPARSE(cusparseDcsrgeam2(handle, rows_, cols_,
    &alpha,
    matA, nnz_,
    values_.data, rowPtr_.data, colInd_.data,
    &beta,
    matB, other.nnz_,
    other.values_.data, other.rowPtr_.data, other.colInd_.data,
    matC,
    result.values_.data, result.rowPtr_.data, result.colInd_.data,
    buffer))

  CHECK_CUSPARSE(cusparseDestroy(handle))
  return result;
}


CuSMat CuSMat::operator*(const CuSMat& other) const {
  CuSMat result(rows_, other.cols_);

  cusparseHandle_t     handle = NULL;
  CHECK_CUSPARSE(cusparseCreate(&handle))

  cusparseOperation_t opA         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t opB         = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpMatDescr_t matA, matB, matC;
  cudaDataType        computeType = CUDA_R_64F;
  double              alpha       = 1.0;
  double              beta        = 0.0;
  void*  dBuffer1    = NULL, *dBuffer2   = NULL;
  size_t bufferSize1 = 0,    bufferSize2 = 0;

  CHECK_CUSPARSE(cusparseCreateCsr(&matA, rows_, cols_, nnz_,
            rowPtr_.data, colInd_.data, values_.data,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
  CHECK_CUSPARSE(cusparseCreateCsr(&matB, other.rows_, other.cols_, other.nnz_,
            other.rowPtr_.data, other.colInd_.data, other.values_.data,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
  CHECK_CUSPARSE(cusparseCreateCsr(&matC, rows_, other.cols_, 0,
            result.rowPtr_.data, NULL, NULL,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
  
  // SpGEMM Computation
  cusparseSpGEMMDescr_t spgemmDesc;
  CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

  // ask bufferSize1 bytes for external memory
  CHECK_CUSPARSE(
      cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, NULL))
  CHECK_CUDA(cudaMalloc((void**) &dBuffer1, bufferSize1))
  // inspect the matrices A and B to understand the memory requirement for
  // the next step
  CHECK_CUSPARSE(
      cusparseSpGEMM_workEstimation(handle, opA, opB,
                                    &alpha, matA, matB, &beta, matC,
                                    computeType, CUSPARSE_SPGEMM_DEFAULT,
                                    spgemmDesc, &bufferSize1, dBuffer1))

  // ask bufferSize2 bytes for external memory
  CHECK_CUSPARSE(
      cusparseSpGEMM_compute(handle, opA, opB,
                              &alpha, matA, matB, &beta, matC,
                              computeType, CUSPARSE_SPGEMM_DEFAULT,
                              spgemmDesc, &bufferSize2, NULL))
  CHECK_CUDA(cudaMalloc((void**) &dBuffer2, bufferSize2))

  // compute the intermediate product of A * B
  CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_DEFAULT,
                                          spgemmDesc, &bufferSize2, dBuffer2))
  // get matrix C non-zero entries nnz
  int64_t num_rows, num_cols, nnz;
  CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &num_rows, &num_cols, &nnz))

  result.colInd_.allocate(nnz);
  result.values_.allocate(nnz);
  result.nnz_ = nnz;

  // update matC with the new pointers
  CHECK_CUSPARSE(
      cusparseCsrSetPointers(matC, result.rowPtr_.data, result.colInd_.data, result.values_.data))

  // if beta != 0, cusparseSpGEMM_copy reuses/updates the values of dC_values

  // copy the final products to the matrix C
  CHECK_CUSPARSE(
      cusparseSpGEMM_copy(handle, opA, opB,
                          &alpha, matA, matB, &beta, matC,
                          computeType, CUSPARSE_SPGEMM_DEFAULT, spgemmDesc))

  CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))

  CHECK_CUDA(cudaFree(dBuffer1))
  CHECK_CUDA(cudaFree(dBuffer2))
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroySpMat(matB))
  CHECK_CUSPARSE(cusparseDestroySpMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))
  return result;
}

CuSMat& CuSMat::operator=(CuSMat&& other) noexcept {
  if (this != &other) {
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
  return *this;
}

CuDMat CuSMat::operator*(const CuDMat& other) const {
  CuDMat result(rows_, other.cols_);

  // CUSPARSE APIs
  double alpha           = 1.0;
  double beta            = 0.0;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseHandle_t     handle = NULL;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;


  CHECK_CUSPARSE(cusparseCreate(&handle))

  CHECK_CUSPARSE(cusparseCreateCsr(&matA, rows_, cols_, nnz_,
            rowPtr_.data, colInd_.data, values_.data,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, other.rows_, other.cols_, other.rows_, other.values_.data, CUDA_R_64F, CUSPARSE_ORDER_COL))
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, rows_, other.cols_, rows_, result.values_.data, CUDA_R_64F, CUSPARSE_ORDER_COL))
  
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
                                  handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute SpMM
  CHECK_CUSPARSE(cusparseSpMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

  // destroy handle
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUSPARSE(cusparseDestroy(handle))
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  return result;
}

Eigen::SparseMatrix<double, Eigen::RowMajor, int> CuSMat::toEigenSparse() const {
  Eigen::SparseMatrix<double, Eigen::RowMajor, int> res;
  convertCuSparseToEigen(rowPtr_.data, colInd_.data, values_.data, nnz_, rows_, cols_, res);
  return res;
}

Eigen::MatrixXd CuDMat::toEigenDense() const {
  Eigen::MatrixXd res(rows_, cols_);
  // CHECK_CUDA(cudaDeviceSynchronize())
  values_.download((double*)res.data());
  return res;
}

void matmul(const CuSMat& A, const CuDMat& B, CuDMat& C) {
  C.resize(A.rows_, B.cols_);
  double alpha           = 1.0;
  double beta            = 0.0;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  cusparseHandle_t     handle = NULL;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;

  CHECK_CUSPARSE(cusparseCreate(&handle))

  CHECK_CUSPARSE(cusparseCreateCsr(&matA, A.rows_, A.cols_, A.nnz_,
            A.rowPtr_.data, A.colInd_.data, A.values_.data,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F))
  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B.rows_, B.cols_, B.rows_, B.values_.data, CUDA_R_64F, CUSPARSE_ORDER_COL))
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, C.rows_, C.cols_, C.rows_, C.values_.data, CUDA_R_64F, CUSPARSE_ORDER_COL))
  
  // allocate an external buffer if needed
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
                                  handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize))

  // execute SpMM
  CHECK_CUSPARSE(cusparseSpMM(handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &alpha, matA, matB, &beta, matC, CUDA_R_64F,
                                  CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))

  // destroy handle
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUSPARSE(cusparseDestroy(handle))
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  // CHECK_CUDA(cudaDeviceSynchronize())
}

void matmul(const CuDMat& A, const CuDMat& B, CuDMat& C) {
  cublasHandle_t handle;
  C.resize(A.rows_, B.cols_);
  assert(A.cols_ == B.rows_);
  int m = A.rows_, n = B.cols_, k = A.cols_;
  double alpha = 1.0;
  double beta = 0.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A.values_.data, m, B.values_.data, k, &beta, C.values_.data, m))
  CHECK_CUBLAS(cublasDestroy(handle))
  CHECK_CUDA(cudaDeviceSynchronize())
}

CuDMat matmul3DM(const CuDMat& A, const CuDMat& B, const CuDMat& C) {
  CuDMat AxB = A * B;
  CuDMat AxBxC = AxB * C;
  return AxBxC;
}

void matmulAtB(const CuDMat& A, const CuDMat& B, CuDMat& AtB) {
  cublasHandle_t handle;
  AtB.resize(A.cols_, B.cols_);
  int m = A.cols_, n = B.cols_, k = B.rows_;
  int lda = A.rows_, ldb = B.rows_, ldc = AtB.rows_;
  double alpha = 1.0;
  double beta = 0.0;
  CHECK_CUBLAS(cublasCreate(&handle))
  CHECK_CUBLAS(cublasDgemm_v2(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, A.values_.data, lda, B.values_.data, ldb, &beta, AtB.values_.data, ldc))
  CHECK_CUBLAS(cublasDestroy(handle))
}

void matmulAtXA(const CuDMat& X, const CuDMat& A, CuDMat& AtXA) {
  AtXA.resize(A.cols_, A.cols_);
  assert(X.cols_ == A.rows_);

  CuDMat AtX; 
  matmulAtB(A, X, AtX);
  matmul(AtX, A, AtXA);
  AtX.release();
}
