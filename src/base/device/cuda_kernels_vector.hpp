#pragma once

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
