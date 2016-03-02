
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
        out[i] = ValueType(0.0);
        for (j = rowPtr[i]; j <rowPtr[i+1]; j++)
        {
            out[i] = out[i] + mat[j]*in[colInd[j]];
        }
    }
}

// Only one thread only for debugging purposes
template <typename ValueType>
__global__ void kernel_csrmatrix_getvalue ( const int i, const int j,
        const int nRows, const int* mRowPtr, const int* mColInd, 
                            ValueType* mData, ValueType* result)
{
    int aux = mRowPtr[i];
    int rowWidth = mRowPtr[i+1] - mRowPtr[i];
    int k;
    for (k = 0; k < rowWidth; k++)
    {
        if(mColInd[aux+k] == j) 
        {
            *result = mData[aux+k];
            break;
        }
        else *result = 0.0;
    
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
