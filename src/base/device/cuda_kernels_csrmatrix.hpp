
#pragma once
#include <stdio.h>

namespace pssolver
{

template <typename ValueType>
__global__ void kernel_csrmatrix_matvec (const int nRows, const int* rowPtr,
        const int* colInd, const ValueType* mat, const ValueType* in, 
        ValueType *out)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if (i < nRows)
    {
        //printf("I = %d\n", i);
        out[i] = ValueType(0.0);
        for (j = rowPtr[i]; j <rowPtr[j+1]; j++)
        {
            out[i] = out[i] + mat[j]*in[colInd[j]];
        }
    }
}

template <typename ValueType>
__global__ void kernel_csrmatrix_matvec_add (const int nRows, const int* rowPtr,
        const int* colInd, const ValueType* mat, const ValueType* in, 
        ValueType *out, const ValueType scalar)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if (i < nRows)
    {
        for (j = rowPtr[i]; j <rowPtr[j+1]; j++)
        {
            out[i] = out[i] + scalar*mat[j]*in[colInd[j]];
        }
    }
}


}
