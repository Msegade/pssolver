#include "DeviceCsrMatrix.hpp"

#include "cuda_utils.h"
#include "cuda_variables.h"

#include "cuda_kernels_csrmatrix.hpp"

#include <iostream>
#include <cassert>

namespace pssolver
{
template <typename ValueType>
DeviceCsrMatrix<ValueType>::DeviceCsrMatrix()
{

}

template <typename ValueType>
DeviceCsrMatrix<ValueType>::~DeviceCsrMatrix()
{
    checkCudaErrors(cudaFree(d_mData));
    checkCudaErrors(cudaFree(d_mColInd));
    checkCudaErrors(cudaFree(d_mRowPtr));

}

template <typename ValueType>
void DeviceCsrMatrix<ValueType>::Allocate(const int nRows, const int nCols, 
                                     const int nnz)
{
    assert(nRows>0 && nCols>0 && nnz>0);
    this->mNRows = nRows; this->mNCols = nCols; this->mNnz = nnz;
    checkCudaErrors(cudaMalloc(&d_mData, nnz*sizeof(ValueType)));
    checkCudaErrors(cudaMalloc(&d_mColInd, nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_mRowPtr, (nRows+1)*sizeof(int)));

}

template <typename ValueType>
void DeviceCsrMatrix<ValueType>::MatVec(BaseVector<ValueType>& invec,
                                        BaseVector<ValueType>& outvec,
                                        ValueType scalar) const
{
    const DeviceVector<ValueType> *cast_in = 
                        dynamic_cast<const DeviceVector<ValueType>*> (&invec);
    DeviceVector<ValueType> *cast_out =
                        dynamic_cast< DeviceVector<ValueType>*> (&outvec);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( this->mNRows / BLOCKSIZE +1);
    kernel_csrmatrix_matvec <<<GridSize, BlockSize>>>( this->mNRows, this->d_mRowPtr,
                this->d_mColInd, this->d_mData, cast_in->d_mData, cast_out->d_mData);
}

template class DeviceCsrMatrix<double>;
template class DeviceCsrMatrix<float>;

}
