#pragma once
#include <stdio.h>

template <typename ValueType>
__global__ void kernel_vector_fill(const int n, ValueType* v, ValueType value)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < n)
        v[ind] = value;
}

template <typename ValueType>
__global__ void kernel_vector_add(const int n,
                                 ValueType* v1, const ValueType* v2)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        v1[ind] = v1[ind] + v2[ind];
}

template <typename ValueType>
__global__ void kernel_vector_add(const int n,
                                 ValueType* out, const ValueType* v1,
                                 const ValueType* v2)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        out[ind] = v1[ind] + v2[ind];
}

template <typename ValueType>
__global__ void kernel_vector_scalar_add(const int n,
                                 ValueType* out, const ValueType* v1,
                                 const ValueType* v2, const ValueType val)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        out[ind] = v1[ind] + val*v2[ind];
}

template <typename ValueType>
__global__ void kernel_vector_substract(const int n,
                                 ValueType* v1, const ValueType* v2)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        v1[ind] = v1[ind] - v2[ind];
}

template <typename ValueType>
__global__ void kernel_vector_substract(const int n,
                                 ValueType* out, const ValueType* v1,
                                 const ValueType* v2)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        out[ind] = v1[ind] - v2[ind];
}

// Element-wise multiply
template <typename ValueType>
__global__ void kernel_vector_multiply(const int n,
                                 ValueType* out, const ValueType* v)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;

    if (ind <n)
        out[ind] = out[ind] * v[ind];
}

template <typename ValueType>
__global__ void kernel_vector_sum_reduce(const int n,  ValueType* gdata)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    int tind = threadIdx.x;

    //Local pointer for this block
    //ValueType *ldata = gdata + blockIdx.x * blockDim.x;

    if (ind >= n ) return;

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>=1)
    {
        //if (tind < stride )
        if ( (tind < stride) && ((ind+stride) < n))
        {
            gdata[ind] += gdata[ind + stride];
        }
        __syncthreads();
    }

    // If I use a local vector index and do
    // gdata[blockIdx.x] = ldata[tind]
    // Causes a race condition in the first element of the array beacause I 
    // read and write to the same address at the same time
    if (tind == 0)
    {
        gdata[blockIdx.x] = gdata[ind];
    }
}

template <typename ValueType>
__global__ void kernel_vector_scalar_multiply(const int n,
                                                ValueType* in, ValueType val)
{
    
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    if (ind < n)
        in[ind] = in[ind]*val;
}

template <typename ValueType>
__global__ void kernel_vector_scalar_multiply(const int n,
                                              const  ValueType* in, ValueType val,
                                                ValueType* out)
{
    
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    if (ind < n)
        out[ind] = in[ind]*val;
}

template <typename ValueType>
__global__ void kernel_vector_add_element(ValueType* vec, const int source, const int dest)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    if (ind == 0)
        vec[dest] = vec[dest] + vec[source]; 
}
