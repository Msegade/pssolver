#include "HostCOOMatrix.hpp"

#include <cassert>
#include <cstring>
#include <iostream>
#include <fstream>

namespace pssolver
{
template <typename ValueType>
HostCOOMatrix<ValueType>::HostCOOMatrix()
{
    mData = new ValueType[0];
    mColInd = new int[0];
    mRowInd = new int[0];
}

template <typename ValueType>
HostCOOMatrix<ValueType>::~HostCOOMatrix()
{
    delete[] mData;
    delete[] mRowInd;
    delete[] mColInd;

}

template <typename ValueType>
void HostCOOMatrix<ValueType>::Allocate(const int nRows,
                                        const int nCols,const int nnz)
{

    assert ( nRows > 0 && nCols > 0 && nnz > 0);
    this->mNRows = nRows; this->mNCols = nCols; this->mNnz = nnz;
    mData = new ValueType[nnz];
    mColInd = new int[nnz];
    mRowInd = new int[nnz];

    memset(mData, 0, nnz*sizeof(ValueType));
    memset(mColInd, 0, nnz*sizeof(int));
    memset(mRowInd, 0, nnz*sizeof(int));
}
template <typename ValueType>
void HostCOOMatrix<ValueType>::ReadFile(const std::string filename)
{
    std::ifstream mFile(filename);
    std::string line;
    std::getline(mFile, line);
    std::cout << line << std::endl;

}

template class HostCOOMatrix<double>;
template class HostCOOMatrix<float>;

}
