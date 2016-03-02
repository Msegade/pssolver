#include "DeviceCsrMatrix.hpp"
#include "../host/HostCsrMatrix.hpp"

#include "../../utils.hpp"

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
    DEBUGLOG(this, "DeviceCsrMatrix::~DeviceCsrMatrix()", "Empty", 2);
    checkCudaErrors(cudaFree(d_mData));
    checkCudaErrors(cudaFree(d_mColInd));
    checkCudaErrors(cudaFree(d_mRowPtr));

}

template <typename ValueType>
void DeviceCsrMatrix<ValueType>::Allocate(const int nRows, const int nCols, 
                                     const int nnz)
{
    DEBUGLOG(this, "DeviceCsrMatrix::Allocate()", "nRows = " << nRows << 
                         " nCols = " << nCols << " nnz = " << nnz, 2);
    assert(nRows>0 && nCols>0 && nnz>0);
    this->mNRows = nRows; this->mNCols = nCols; this->mNnz = nnz;
    checkCudaErrors(cudaMalloc(&d_mData, nnz*sizeof(ValueType)));
    checkCudaErrors(cudaMalloc(&d_mColInd, nnz*sizeof(int)));
    checkCudaErrors(cudaMalloc(&d_mRowPtr, (nRows+1)*sizeof(int)));

}
template <typename ValueType>
void DeviceCsrMatrix<ValueType>::Print(std::ostream& os) 
{
    HostCsrMatrix<ValueType> tmp;
    this->CopyToHost(tmp);
    tmp.Print(os);
}


template <typename ValueType>
void DeviceCsrMatrix<ValueType>::CopyFromHost(
                                       const BaseMatrix<ValueType> &hostMatrix)
{
    DEBUGLOG(this, "DeviceCsrMatrix::CopyFromHost()", "Mat = " << &hostMatrix, 2);
    if ( (this->mNRows != hostMatrix.GetNRows()) || 
        (this->mNCols != hostMatrix.GetNCols()) ||
        (this->mNnz   != hostMatrix.GetNnz()) )
    {
        this->Allocate(hostMatrix.GetNRows(), hostMatrix.GetNCols(), hostMatrix.GetNnz());
    }
    const HostCsrMatrix<ValueType> *cast_host = 
                dynamic_cast<const HostCsrMatrix<ValueType>*> (&hostMatrix);

    checkCudaErrors(cudaMemcpy(d_mData, cast_host->mData, this->mNnz*sizeof(ValueType),
                            cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mColInd, cast_host->mColInd, this->mNnz*sizeof(int),
                            cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_mRowPtr, cast_host->mRowPtr, (this->mNRows+1)*sizeof(int),
                            cudaMemcpyHostToDevice));


}

template <typename ValueType>
void DeviceCsrMatrix<ValueType>::CopyFromDevice(
                                       const BaseMatrix<ValueType> &deviceMatrix)
{
    DEBUGLOG(this, "DeviceCsrMatrix::CopyFromDevice()", "Mat = " << &deviceMatrix, 2);
    this->Allocate(deviceMatrix.GetNRows(), deviceMatrix.GetNCols(), deviceMatrix.GetNnz());
    const DeviceCsrMatrix<ValueType> *cast_device = 
                dynamic_cast<const DeviceCsrMatrix<ValueType>*> (&deviceMatrix);
    checkCudaErrors(cudaMemcpy(d_mData, cast_device->d_mData, this->mNnz*sizeof(ValueType),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_mColInd, cast_device->d_mColInd, this->mNnz*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(d_mRowPtr, cast_device->d_mRowPtr, (this->mNRows+1)*sizeof(int),
                            cudaMemcpyDeviceToDevice));

}

template <typename ValueType>
void DeviceCsrMatrix<ValueType>::CopyToHost(
                                       BaseMatrix<ValueType> &hostMatrix) const
{
    DEBUGLOG(this, "DeviceCsrMatrix::CopyToHost()", "Mat = " << &hostMatrix, 2);
    if ( (this->mNRows != hostMatrix.GetNRows()) || 
        (this->mNCols != hostMatrix.GetNCols()) ||
        (this->mNnz   != hostMatrix.GetNnz()) )
    {
        hostMatrix.Allocate(this->GetNRows(), this->GetNCols(), this->GetNnz());
    }
    const HostCsrMatrix<ValueType> *cast_host = 
                dynamic_cast<const HostCsrMatrix<ValueType>*> (&hostMatrix);
    checkCudaErrors(cudaMemcpy(cast_host->mData, d_mData, this->mNnz*sizeof(ValueType),
                            cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cast_host->mColInd, d_mColInd,  this->mNnz*sizeof(int),
                            cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(cast_host->mRowPtr, d_mRowPtr, (this->mNRows+1)*sizeof(int),
                            cudaMemcpyDeviceToHost));

}

template <typename ValueType>
void DeviceCsrMatrix<ValueType>::CopyToDevice(
                                       BaseMatrix<ValueType> &deviceMatrix) const
{
    DEBUGLOG(this, "DeviceCsrMatrix::CopyToDevice()", "Mat = " << &deviceMatrix, 2);
    deviceMatrix.Allocate(this->GetNRows(), this->GetNCols(), this->GetNnz());
    const DeviceCsrMatrix<ValueType> *cast_device = 
                dynamic_cast<const DeviceCsrMatrix<ValueType>*> (&deviceMatrix);
    checkCudaErrors(cudaMemcpy(cast_device->d_mData, d_mData, this->mNnz*sizeof(ValueType),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(cast_device->d_mColInd, d_mColInd, this->mNnz*sizeof(int),
                            cudaMemcpyDeviceToDevice));
    checkCudaErrors(cudaMemcpy(cast_device->d_mRowPtr, d_mRowPtr, (this->mNRows+1)*sizeof(int),
                            cudaMemcpyDeviceToDevice));

}

template <typename ValueType>
ValueType DeviceCsrMatrix<ValueType>::Read(int i, int j) const
{
    DEBUGLOG(this, "DeviceCsrMatrix::Read()", "i = " << i << " j = " << j, 2);
    assert (i >=0 && j>=0 && i < this->mNRows && j < this->mNCols);
    // Launch a single therad only to get the value
    ValueType result = 0.0;
    ValueType* d_result;
    checkCudaErrors(cudaMalloc(&d_result, sizeof(ValueType)));
    kernel_csrmatrix_getvalue <<<1, 1>>>( i, j, this->mNRows, 
            this->d_mRowPtr, this->d_mColInd, this->d_mData, d_result);
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );
    checkCudaErrors(cudaMemcpy(&result, d_result, sizeof(ValueType),
                            cudaMemcpyDeviceToHost));

    return result;
    

}

template <typename ValueType>
void DeviceCsrMatrix<ValueType>::MatVec(BaseVector<ValueType>& invec,
                                        BaseVector<ValueType>& outvec,
                                        ValueType scalar) const
{
    DEBUGLOG(this, "DeviceCsrMatrix::MatVec()", "InVec = " << &invec
                    << " OutVec = " << &outvec << " Scalar = " << scalar, 2);
    const DeviceVector<ValueType> *cast_in = 
                        dynamic_cast<const DeviceVector<ValueType>*> (&invec);
    DeviceVector<ValueType> *cast_out =
                        dynamic_cast< DeviceVector<ValueType>*> (&outvec);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( this->mNRows / BLOCKSIZE +1);
    kernel_csrmatrix_matvec <<<GridSize, BlockSize>>>( this->mNRows, this->d_mRowPtr,
                this->d_mColInd, this->d_mData, cast_in->d_mData, cast_out->d_mData);
    checkCudaErrors( cudaPeekAtLastError() );
    checkCudaErrors( cudaDeviceSynchronize() );
}

template class DeviceCsrMatrix<double>;
template class DeviceCsrMatrix<float>;

}
