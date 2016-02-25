#include "DeviceVector.hpp"
#include "../host/HostVector.hpp"

#include "../../utils.hpp"

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
DeviceVector<ValueType>::DeviceVector(const DeviceVector<ValueType>& v)
{
    DEBUGLOG(this, "DeviceVector::DeviceVector()", "Vec = " << &v, 2);
    this->Allocate(v.mSize);
    this->CopyFromDevice(v);
}

template <typename ValueType>
DeviceVector<ValueType>::~DeviceVector()
{
    DEBUGLOG(this, "DeviceVector::~DeviceVector()", "Empty", 2);
    checkCudaErrors(cudaFree(d_mData)); 
}

template <typename ValueType>
void DeviceVector<ValueType>::Allocate (const int size)
{
    DEBUGLOG(this, "DeviceVector::Allocate()", "size = " << size, 2);
    assert(size >= 0);
    mSize = size;
    checkCudaErrors(cudaMalloc(&d_mData, mSize*sizeof(double)));
}

template <typename ValueType>
void DeviceVector<ValueType>::SetVal (const ValueType val)
{
    DEBUGLOG(this, "DeviceVector::SetVal()", "val = " << val, 2);
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_fill <<<GridSize, BlockSize>>>(mSize, d_mData, val);
}

template <typename ValueType>
ValueType DeviceVector<ValueType>::Read(const int i) const
{
    ValueType val;
    checkCudaErrors(cudaMemcpy(&val, &(d_mData[i]), sizeof(ValueType),
                    cudaMemcpyDeviceToHost));
    return val;
}

template <typename ValueType>
void DeviceVector<ValueType>::Print(std::ostream& os)
{
}


template <typename ValueType>
void DeviceVector<ValueType>::CopyFromHost(const BaseVector<ValueType>& src)
{
    DEBUGLOG(this, "DeviceVector::CopyFromHost()", "Vec = " << &src, 2);
    this->Allocate(src.GetSize());
    const HostVector<ValueType> *cast_vec; 
    cast_vec = dynamic_cast<const HostVector<ValueType>*> (&src);

    checkCudaErrors(cudaMemcpy(d_mData, cast_vec->mData, mSize*sizeof(ValueType),
                    cudaMemcpyHostToDevice));

}

template <typename ValueType>
void DeviceVector<ValueType>::CopyFromDevice(const BaseVector<ValueType>& src)
{
    DEBUGLOG(this, "DeviceVector::CopyFromDevice()", "Vec = " << &src, 2);
    this->Allocate(src.GetSize());
    const DeviceVector<ValueType> *cast_vec; 
    cast_vec = dynamic_cast<const DeviceVector<ValueType>*> (&src);

    checkCudaErrors(cudaMemcpy(d_mData, cast_vec->d_mData, mSize*sizeof(ValueType),
                    cudaMemcpyDeviceToDevice));

}

template <typename ValueType>
void DeviceVector<ValueType>::CopyToHost(BaseVector<ValueType>& dst) const
{
    DEBUGLOG(this, "DeviceVector::CopyToHost()", "Vec = " << &dst, 2);
    dst.Allocate(this->GetSize());
    const HostVector<ValueType> *cast_vec; 
    cast_vec = dynamic_cast<const HostVector<ValueType>*> (&dst);

    checkCudaErrors(cudaMemcpy(cast_vec->mData, d_mData, mSize*sizeof(ValueType),
                    cudaMemcpyDeviceToHost));

}

template <typename ValueType>
void DeviceVector<ValueType>::CopyToDevice(BaseVector<ValueType>& dst) const
{
    DEBUGLOG(this, "DeviceVector::CopyToDevice()", "Vec = " << &dst, 2);
    dst.Allocate(this->GetSize());
    const DeviceVector<ValueType> *cast_vec; 
    cast_vec = dynamic_cast<const DeviceVector<ValueType>*> (&dst);

    checkCudaErrors(cudaMemcpy(cast_vec->d_mData, d_mData, mSize*sizeof(ValueType),
                    cudaMemcpyDeviceToDevice));

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
    DeviceVector<ValueType> aux(*this);
    double result = 0.0;
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_multiply <<<GridSize, BlockSize>>> ( mSize, aux.d_mData,
                                                    d_mData);

    result = aux.SumReduce();

    return sqrt((double)result);

}

template <typename ValueType>
ValueType DeviceVector<ValueType>::Dot(const BaseVector<ValueType>& otherVector)
{
    const DeviceVector<ValueType> *cast_v = 
        dynamic_cast<const DeviceVector<ValueType>*> (&otherVector);

    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    DeviceVector<ValueType> aux(*this);
    kernel_vector_multiply <<<GridSize, BlockSize>>> ( mSize, aux.d_mData,
                                                    cast_v->d_mData);
    ValueType result = aux.SumReduce();

    return result;
       
    
}

template <typename ValueType>
void DeviceVector<ValueType>::ScalarMul(const ValueType& val)
{
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_scalar_multiply <<<GridSize, BlockSize>>> (mSize, d_mData, val);
        
}
template <typename ValueType>
ValueType DeviceVector<ValueType>::SumReduce(void)
{
    dim3 BlockSize(BLOCKSIZE);
    dim3 GridSize( mSize / BLOCKSIZE +1);
    kernel_vector_sum_reduce <<<GridSize, BlockSize>>> ( mSize, d_mData);

    //If the grid size is odd, we get at an odd number of elements at the beginning of the
    // array that we need to sum, and the algorithm kernel_sum_reduce_onevector
    // only reduces an even number of elements
    ValueType result=0.0;
    if ( GridSize.x % 2 != 0)
    {
        // We change the element to 0 and increase the size of the grid by one
        // to get an even number of elements
        ValueType zero = 0.0;
        checkCudaErrors(cudaMemcpy(&d_mData[GridSize.x], &zero, sizeof(double), cudaMemcpyHostToDevice));
        BlockSize = GridSize.x + 1;
        GridSize = 1;
        kernel_vector_sum_reduce <<<GridSize, BlockSize>>> (mSize, d_mData);
    }
    BlockSize = GridSize;
    GridSize = 1;
    kernel_vector_sum_reduce <<<GridSize, BlockSize>>> (mSize, d_mData);

    checkCudaErrors(cudaMemcpy( &result, d_mData, sizeof(double), cudaMemcpyDeviceToHost));

    return result;
}

template class DeviceVector<double>;
template class DeviceVector<float>;
template class DeviceVector<int>;

}
