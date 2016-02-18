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
__global__ void kernel_vector_sum_reduce(const int n,  ValueType* in)
{
    int ind = threadIdx.x + blockDim.x * blockIdx.x;
    int tind = threadIdx.x;

        for (unsigned int s = blockDim.x / 2; s > 0; s>>=1 )
        {
            // If ind+s is greater than n It takes garbage
            if (tind < s && (ind+s) < n)
            {
                //in[ind] = 1.0;
                in[ind] += in[ind + s];
            }
        }

        if (tind == 0)
        {
            in[blockIdx.x] = in[ind];
        }
}

