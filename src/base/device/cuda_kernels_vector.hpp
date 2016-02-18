#pragma once

template <typename ValueType>
__global__ void kernel_fill_vector(const int n, ValueType* v, ValueType value)
{
    int ind = blockIdx.x*blockDim.x + threadIdx.x;
    if (ind < n)
        v[ind] = value;
}
