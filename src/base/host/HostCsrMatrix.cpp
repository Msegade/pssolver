#include "HostCsrMatrix.hpp"

#include <cassert>
#include <cstring>
#include <iostream>

namespace pssolver
{
template <typename ValueType>
HostCsrMatrix<ValueType>::HostCsrMatrix()
{
    mData = new ValueType[0];
    mColInd = new int[0];
    mRowPtr = new int[0];
}

template <typename ValueType>
HostCsrMatrix<ValueType>::~HostCsrMatrix()
{
    delete[] mData;
    delete[] mRowPtr;
    delete[] mColInd;

}

template <typename ValueType>
void HostCsrMatrix<ValueType>::Allocate(const int nRows,
                                        const int nCols,const int nnz)
{

    assert ( nRows > 0 && nCols > 0 && nnz > 0);
    this->mNRows = nRows; this->mNCols = nCols; this->mNnz = nnz;
    mData = new ValueType[nnz];
    mColInd = new int[nnz];
    mRowPtr = new int[nRows+1];

    memset(mData, 0, nnz*sizeof(ValueType));
    memset(mColInd, 0, nnz*sizeof(int));
    memset(mRowPtr, 0, (nRows+1)*sizeof(int));
}

template <typename ValueType>
void HostCsrMatrix<ValueType>::Print(std::ostream& os)
{
    os << "Data" << "\t" << "Col Index" << "\t" << 
                                                    "Row Ptr" << std::endl;
    for (int i= 0; i< this->mNnz; i++)
    {
        if (i <= this->mNRows)
        {
        os << mData[i] << "\t" << mColInd[i] << "\t" << 
                                                    mRowPtr[i] << std::endl;

        }
        else 
        os << mData[i] << "\t" << mColInd[i] << "\t" << std::endl;

    }
}

template class HostCsrMatrix<double>;
template class HostCsrMatrix<float>;

}
