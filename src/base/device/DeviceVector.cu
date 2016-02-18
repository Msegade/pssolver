#include "DeviceVector.hpp"
#include "../host/HostVector.hpp"

#include "cuda_utils.h"

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
void DeviceVector<ValueType>::CopyFrom(const BaseVector<ValueType>& src)
{
    const HostVector<ValueType> *cast_vec; 
    cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src);

    checkCudaErrors(cudaMemcpy(d_mData, cast_vec->mData, mSize*sizeof(double),
                    cudaMemcpyHostToDevice));

}

template class DeviceVector<double>;
template class DeviceVector<float>;
template class DeviceVector<int>;

}
