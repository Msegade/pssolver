#include "DeviceVector.hpp"
#include "../host/HostVector.hpp"

#include "cuda_kernels_vector.hpp"
#include "cuda_utils.h"
#include "cuda_variables.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <cmath>

namespace pssolver
{

template <typename ValueType>
DeviceVector<ValueType>::DeviceVector()
{
        
}

template <typename ValueType>
DeviceVector<ValueType>::~DeviceVector()
{
    checkCudaErrors(cudaFree(d_mData)); 
}

template <typename ValueType>
void DeviceVector<ValueType>::Allocate (const int size)
{
    assert(size > 0);
    mSize = size;
    checkCudaErrors(cudaMalloc(&d_mData, mSize*sizeof(double)));
}

template <typename ValueType>
void DeviceVector<ValueType>::SetVal (const ValueType val)
{
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_fill <<<GridSize, BlockSize>>>(mSize, d_mData, val);
}

template <typename ValueType>
void DeviceVector<ValueType>::CopyFrom(const BaseVector<ValueType>& src)
{
    const HostVector<ValueType> *cast_vec; 
    cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src);

    checkCudaErrors(cudaMemcpy(d_mData, cast_vec->mData, mSize*sizeof(double),
                    cudaMemcpyHostToDevice));

}

template <typename ValueType>
void DeviceVector<ValueType>::Add(
                        const BaseVector<ValueType> &otherVector)
{
    const DeviceVector<ValueType> *cast_vec = 
        dynamic_cast<const DeviceVector<ValueType>*> (&otherVector);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_add <<<GridSize, BlockSize>>>
                                    (mSize, d_mData, cast_vec->d_mData);

}

template <typename ValueType>
void DeviceVector<ValueType>::Add(
                        const BaseVector<ValueType> &v1,
                        const BaseVector<ValueType> &v2)

{
    const DeviceVector<ValueType> *cast_v1 = 
        dynamic_cast<const DeviceVector<ValueType>*> (&v1);
    const DeviceVector<ValueType> *cast_v2 = 
        dynamic_cast<const DeviceVector<ValueType>*> (&v2);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_add <<<GridSize, BlockSize>>>
                                    (mSize, d_mData,
                                     cast_v1->d_mData,
                                     cast_v2->d_mData);

}

template <typename ValueType>
double DeviceVector<ValueType>::Norm(void) const
{
    double result = 0.0;
    return result;

}

template class DeviceVector<double>;
template class DeviceVector<float>;
template class DeviceVector<int>;

}
